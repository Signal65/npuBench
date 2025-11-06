#!/usr/bin/env python3
"""
Python benchmark runner for npuBench (OpenAI-compatible servers).

Features:
- Loads prompts from a local JSON/JSONL file (default: prompts.json)
- Per-prompt overrides for generation params (temperature, top_p, seed,
  max_output_tokens, runs_per_prompt, tokenizer_model_id, force_tokenizer)
- MLPerf-style warmup: first run per prompt is marked is_warmup=true
- Pauses between runs (default 5s) and between prompts (default 60s)
- Model preflight check against /v1/models (case-insensitive)
- Streams tokens to measure TTFT and total latency; collects usage if provided
- Heuristics to flag bad outputs and missing metrics
- Writes results to CSV

Example:
  python tools/bench.py \
    --base-url http://127.0.0.1:18181/v1/ \
    --model Llama3.2-3B-NPU-Turbo \
    --backend-name Nexa \
    --prompts-file prompts.json \
    --out-csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
import platform
import socket
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

try:
    import requests  # type: ignore
except Exception as exc:  # pragma: no cover - baseline runtime dependency
    raise SystemExit("The requests package is required. Install with 'pip install requests'.") from exc

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore


# --------------------------- I/O and config ---------------------------


def read_json_or_jsonl(path: str) -> Any:
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def api_root(base_url: str) -> str:
    # Do not auto-append "/v1"; expect caller to provide full base including version.
    return base_url.rstrip("/")


def get_models(base_url: str, timeout: int, debug: bool = False) -> List[str]:
    root = api_root(base_url)
    url = root + "/models"
    try:
        if debug:
            print(f"[net] GET {url} timeout={timeout}s (base={base_url})")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        models = [str(x.get("id", "")).lower() for x in data.get("data", [])]
        if debug:
            print(f"[net] /v1/models -> {len(models)} models")
        return models
    except Exception as exc:
        if debug:
            print(f"[net] /v1/models failed: {type(exc).__name__}: {exc}")
        return []


def merge_params(cli: Dict[str, Any], file_defaults: Dict[str, Any], prompt_overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = {}
    for key in [
        "temperature",
        "top_p",
        "seed",
        "max_output_tokens",
        "tokenizer_model_id",
        "force_tokenizer",
    ]:
        if key in prompt_overrides and prompt_overrides[key] is not None:
            merged[key] = prompt_overrides[key]
        elif key in file_defaults and file_defaults[key] is not None:
            merged[key] = file_defaults[key]
        elif key in cli and cli[key] is not None:
            merged[key] = cli[key]
        else:
            merged[key] = None
    return merged


# --------------------------- Tokenizer helpers ---------------------------


def count_tokens_with_transformers(tokenizer_id: Optional[str], text: str) -> Optional[int]:
    if not tokenizer_id or AutoTokenizer is None:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=False)
        return len(tok.encode(text))
    except Exception:
        return None


# --------------------------- Streaming chat completion ---------------------------


@dataclass
class StreamMetrics:
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    gen_tokens_per_s: Optional[float] = None
    last_stream_usage: Optional[dict] = None


def call_chat_stream(
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
    seed: Optional[int],
    timeout: int,
    debug_usage: bool,
) -> Tuple[str, StreamMetrics]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": True,
        # Request usage objects in the streamed packets if the server supports it
        "stream_options": {"include_usage": True},
    }

    # Remove None values (some servers are strict)
    body = {k: v for k, v in body.items() if v is not None}

    root = api_root(base_url)
    url = root + "/chat/completions"
    t0 = time.time()
    try:
        payload_len = len(json.dumps(body, ensure_ascii=False))
    except Exception:
        payload_len = -1
    if debug_usage:
        print(f"[net] POST {url} timeout={timeout}s stream=True payload_bytes={payload_len} (base={base_url})")
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout, stream=True)
        if debug_usage:
            print(f"[net] -> status={resp.status_code}")
        resp.raise_for_status()
    except Exception as exc:
        if debug_usage:
            print(f"[net] request error: {type(exc).__name__}: {exc}")
        raise

    text_parts: List[str] = []
    seen_first_token = False
    metrics = StreamMetrics()
    last_usage: Optional[dict] = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="ignore")
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        # capture usage if present on streamed packets
        if isinstance(obj, dict) and obj.get("usage"):
            last_usage = obj.get("usage")

        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        token = delta.get("content")
        if token is not None:
            if not seen_first_token:
                metrics.ttft_ms = (time.time() - t0) * 1000.0
                seen_first_token = True
            text_parts.append(token)

    text = "".join(text_parts)
    metrics.total_ms = (time.time() - t0) * 1000.0

    # Try to get usage via non-stream call if missing
    if debug_usage or not last_usage:
        try:
            no_stream_body = dict(body)
            no_stream_body.pop("stream", None)
            if debug_usage:
                print(f"[net] POST {url} (non-stream) timeout={timeout}s")
            r2 = requests.post(url, headers=headers, json=no_stream_body, timeout=timeout)
            if r2.ok:
                j2 = r2.json()
                last_usage = (j2.get("usage") or last_usage)
                if debug_usage:
                    print("[net] non-stream usage received")
        except Exception as exc:
            if debug_usage:
                print(f"[net] non-stream usage failed: {type(exc).__name__}: {exc}")
            pass

    metrics.last_stream_usage = last_usage
    if last_usage:
        metrics.prompt_tokens = last_usage.get("prompt_tokens")
        metrics.completion_tokens = last_usage.get("completion_tokens")
        gen_ms = max(metrics.total_ms or 0.0, 0.001) - max(metrics.ttft_ms or 0.0, 0.0)
        if gen_ms > 0 and metrics.completion_tokens is not None:
            metrics.gen_tokens_per_s = float(metrics.completion_tokens) / (gen_ms / 1000.0)

    return text, metrics


# --------------------------- Output heuristics ---------------------------


def flag_output(text: str, metrics: StreamMetrics) -> Tuple[bool, List[str]]:
    flags: List[str] = []
    ok = True

    if text is None or len(text.strip()) == 0:
        ok = False
        flags.append("empty_output")
    if len(text.strip()) < 5:
        ok = False
        flags.append("too_short")
    lower = text.lower()
    if "error:" in lower or "traceback" in lower:
        ok = False
        flags.append("error_text")
    # naive repetition heuristic
    words = lower.split()
    if len(words) >= 8 and len(set(words)) / max(1, len(words)) < 0.3:
        ok = False
        flags.append("repetition")
    if metrics.gen_tokens_per_s is None:
        ok = False
        flags.append("missing_gen_tps")
    return ok, flags


# --------------------------- Main ---------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Python benchmark runner for OpenAI-compatible servers")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend-name", default="")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--prompts-file", default="prompts.json")
    ap.add_argument("--out-csv", default=None, help="Output CSV path; if omitted, writes to results/bench_<timestamp>.csv")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-output-tokens", type=int)
    ap.add_argument("--runs-per-prompt", type=int, default=4, help="Number of runs to execute per prompt")
    ap.add_argument("--pause-between-runs-sec", type=int, default=5)
    ap.add_argument("--pause-between-prompts-sec", type=int, default=60)
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--exclude-warmup-from-csv", action="store_true")
    ap.add_argument("--debug-usage", action="store_true")
    ap.add_argument("--tokenizer-model-id")
    ap.add_argument("--force-tokenizer", action="store_true")
    ap.add_argument("--debug-network", action="store_true", help="Verbose network debug logs")
    args = ap.parse_args()

    try:
        prompts_data = read_json_or_jsonl(args.prompts_file)
    except Exception as exc:
        print(f"Failed to read prompts file {args.prompts_file}: {exc}", file=sys.stderr)
        return 2

    # normalize structure
    if isinstance(prompts_data, dict):
        prompts = prompts_data.get("prompts") or prompts_data.get("items") or prompts_data.get("bench_prompts")
        file_defaults = {
            "temperature": prompts_data.get("temperature"),
            "top_p": prompts_data.get("top_p"),
            "seed": prompts_data.get("seed"),
            "max_output_tokens": prompts_data.get("max_output_tokens"),
            "tokenizer_model_id": prompts_data.get("tokenizer_model_id"),
            "force_tokenizer": prompts_data.get("force_tokenizer"),
        }
    else:
        prompts = prompts_data
        file_defaults = {}

    if not prompts:
        print(f"No prompts found in {args.prompts_file}", file=sys.stderr)
        return 2

    # Preflight model
    available = get_models(args.base_url, timeout=args.timeout_sec, debug=args.debug_network)
    if available and args.model.lower() not in available:
        print(f"ERROR: model '{args.model}' not found in /v1/models; aborting.", file=sys.stderr)
        return 3

    cli_defaults = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_output_tokens": args.max_output_tokens,
        "tokenizer_model_id": args.tokenizer_model_id,
        "force_tokenizer": args.force_tokenizer,
    }

    rows: List[Dict[str, Any]] = []
    # System metadata captured once per run
    from datetime import datetime as _dt
    bench_timestamp = _dt.now().isoformat(timespec='seconds')
    hostname = socket.gethostname()
    os_version = platform.platform()
    run_index = 0
    total_runs = 0
    # compute total runs for progress
    for p in prompts:
        total_runs += int(args.runs_per_prompt)

    prompt_idx = 0
    for p in prompts:
        pid = p.get("id") or p.get("name") or f"prompt_{prompt_idx}"
        merged = merge_params(cli_defaults, file_defaults, p)
        rpp = int(args.runs_per_prompt)
        temperature = merged.get("temperature")
        top_p = merged.get("top_p")
        seed = merged.get("seed")
        max_tokens = merged.get("max_output_tokens")

        # build messages
        messages = p.get("messages")
        if not messages:
            sys_text = p.get("system") or ""
            user_text = p.get("user") or p.get("prompt") or ""
            messages = []
            if sys_text:
                messages.append({"role": "system", "content": sys_text})
            messages.append({"role": "user", "content": user_text})

        # prompt-level status
        print(f"[prompt {prompt_idx + 1}/{len(prompts)}] {pid} | runs={rpp}")

        for i in range(1, rpp + 1):
            run_index += 1
            is_warmup = (i == 1)

            # run-level start status
            label = " warmup" if is_warmup else ""
            print(f"[run {run_index}/{total_runs}] {pid} (run {i}/{rpp}{label}) ...")

            try:
                output, metrics = call_chat_stream(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    timeout=args.timeout_sec,
                    debug_usage=(args.debug_usage or args.debug_network),
                )
            except Exception as exc:
                output = ""
                metrics = StreamMetrics()
                metrics.total_ms = None
                ok = False
                flags = ["request_error"]
                print(f"[run {run_index}/{total_runs}] {pid} request_error: {type(exc).__name__}: {exc}")
                row = {
                    "timestamp": bench_timestamp,
                    "hostname": hostname,
                    "os_version": os_version,
                    "base_url": args.base_url,
                    "prompt_id": pid,
                    "run_index": run_index,
                    "is_warmup": is_warmup,
                    "ttft_ms": metrics.ttft_ms,
                    "total_ms": metrics.total_ms,
                    "gen_tokens_per_s": metrics.gen_tokens_per_s,
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "output": output,
                    "output_ok": ok,
                    "output_flags": ",".join(flags),
                    "model": args.model,
                    "backend_name": args.backend_name,
                }
                rows.append(row)
                # pause between runs (even on failure)
                if i < rpp:
                    time.sleep(args.pause_between_runs_sec)
                continue

            ok, flags = flag_output(output, metrics)

            if args.debug_usage:
                print(
                    f"[usage] ttft_ms={metrics.ttft_ms} total_ms={metrics.total_ms} "
                    f"prompt_tokens={metrics.prompt_tokens} completion_tokens={metrics.completion_tokens} "
                    f"gen_tps={(metrics.gen_tokens_per_s)}"
                )
                if metrics.last_stream_usage is not None:
                    print("[usage] last stream usage:", json.dumps(metrics.last_stream_usage, ensure_ascii=False))

            # Fallback: compute completion token count via tokenizer if server omitted usage
            if (metrics.gen_tokens_per_s is None) and output and (args.force_tokenizer or args.tokenizer_model_id):
                ctoks = count_tokens_with_transformers(args.tokenizer_model_id, output)
                if ctoks is not None:
                    metrics.completion_tokens = ctoks
                    gen_ms = max(metrics.total_ms or 0.0, 0.001) - max(metrics.ttft_ms or 0.0, 0.0)
                    if gen_ms > 0:
                        metrics.gen_tokens_per_s = float(ctoks) / (gen_ms / 1000.0)

            # run-level completion status
            ttft = f"{metrics.ttft_ms:.1f}" if metrics.ttft_ms is not None else "na"
            gtps = f"{metrics.gen_tokens_per_s:.3f}" if metrics.gen_tokens_per_s is not None else "na"
            end_label = " ok" if ok else f" flags={','.join(flags)}"
            print(f"[run {run_index}/{total_runs}] {pid} -> TTFT {ttft} ms, gen_tps {gtps}{end_label}")

            row = {
                "timestamp": bench_timestamp,
                "hostname": hostname,
                "os_version": os_version,
                "base_url": args.base_url,
                "prompt_id": pid,
                "run_index": run_index,
                "is_warmup": is_warmup,
                "ttft_ms": round(metrics.ttft_ms or 0.0, 3) if metrics.ttft_ms is not None else None,
                "total_ms": round(metrics.total_ms or 0.0, 3) if metrics.total_ms is not None else None,
                "gen_tokens_per_s": round(metrics.gen_tokens_per_s or 0.0, 4) if metrics.gen_tokens_per_s is not None else None,
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "output": output,
                "output_ok": ok,
                "output_flags": ",".join(flags),
                "model": args.model,
                "backend_name": args.backend_name,
            }
            rows.append(row)

            # pause between runs of same prompt (skip after last run)
            if i < rpp:
                time.sleep(args.pause_between_runs_sec)

        prompt_idx += 1
        # pause between prompts (skip after last prompt)
        if prompt_idx < len(prompts):
            # Pause between prompts only for prompts that have multiple runs by default.
            # Per-prompt override via pause_after_sec always takes precedence.
            p_pause = p.get("pause_after_sec")
            if isinstance(p_pause, (int, float)):
                pause_sec = int(p_pause)
            else:
                pause_sec = int(args.pause_between_prompts_sec) if rpp > 1 else 0
            if pause_sec > 0:
                print(f"[pause] between prompts: {pause_sec}s")
                time.sleep(pause_sec)

    # write CSV
    # Determine output path: if caller gave only a filename (no directory),
    # place it under ./results and append a timestamp to avoid overwrites.
    out_csv_arg = args.out_csv
    if not out_csv_arg:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath(os.path.join(os.getcwd(), "results"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"bench_{ts}.csv")
    else:
        dir_part = os.path.dirname(out_csv_arg)
        if dir_part:
            out_path = os.path.abspath(out_csv_arg)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        else:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(os.path.basename(out_csv_arg))[0] or "bench"
            out_dir = os.path.abspath(os.path.join(os.getcwd(), "results"))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base}_{ts}.csv")
    # optionally exclude warmup
    out_rows = [r for r in rows if not (args.exclude_warmup_from_csv and r.get("is_warmup"))]
    if not out_rows:
        print("No rows to write (possibly excluded all warmups).", file=sys.stderr)

    fieldnames = [
        "timestamp",
        "hostname",
        "os_version",
        "base_url",
        "prompt_id",
        "run_index",
        "is_warmup",
        "ttft_ms",
        "total_ms",
        "gen_tokens_per_s",
        "prompt_tokens",
        "completion_tokens",
        "output",
        "output_ok",
        "output_flags",
        "model",
        "backend_name",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


