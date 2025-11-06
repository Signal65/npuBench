#!/usr/bin/env python3
"""
Simple single-machine orchestrator for npuBench.

Runs a matrix of jobs defined in a YAML or JSON plan file. Each job:
  1) Preflights the model against /v1/models (case-insensitive).
  2) Invokes bench.ps1 to produce a CSV.
  3) Invokes tools/evaluate.py to grade results.
  4) Writes artifacts under runs/<ts>_<name>_<model>/ and updates an index.

Usage:
  python tools/orchestrate.py --plan tools/plan.yaml

Dependencies:
  - JSON plans: no extra deps
  - YAML plans: PyYAML (pip install pyyaml)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List
import ast
import re

try:
    import requests  # type: ignore
except Exception:
    requests = None  # noqa: E305


def read_plan(path: str) -> Dict[str, Any]:
    # Normalize path and be tolerant of accidental trailing dots/spaces
    normalized = path.strip().rstrip(".")
    with open(normalized, "r", encoding="utf-8") as f:
        text = f.read()
    lower = normalized.lower()
    # Prefer parser by extension, but fall back gracefully
    if lower.endswith(".yaml") or lower.endswith(".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "YAML plan given but PyYAML is not installed. Install with 'pip install pyyaml' or provide a JSON plan."
            ) from exc
        return yaml.safe_load(text) or {}
    # Heuristic: try JSON first; on failure try YAML if available
    try:
        return json.loads(text or "{}")
    except Exception:
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text) or {}
        except Exception:
            raise


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_safe_segment(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", text)


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def api_root(base_url: str) -> str:
    # Do not auto-append "/v1"; expect caller to include it.
    return base_url.rstrip("/")


def get_models(base_url: str, debug: bool = False) -> List[str]:
    if not requests:
        return []
    root = api_root(base_url)
    url = root + "/models"
    try:
        if debug:
            print(f"[net] GET {url} timeout=10s (base={base_url})")
        r = requests.get(url, timeout=10)
        if debug:
            print(f"[net] -> status={r.status_code}")
        r.raise_for_status()
        data = r.json()
        items = data.get("data", [])
        return [str(item.get("id", "")).lower() for item in items]
    except Exception as exc:
        if debug:
            print(f"[net] /v1/models failed: {type(exc).__name__}: {exc}")
        return []


def run_process(cmd: List[str], log_path: str | None = None, env: Dict[str, str] | None = None) -> int:
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    # Force unbuffered output for Python children so we can stream logs in real time
    proc_env.setdefault("PYTHONUNBUFFERED", "1")
    # If logging to file, tee stdout to console and file in real time
    if log_path:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=proc_env,
            text=True,
            bufsize=1,
        ) as p:
            ensure_dir(os.path.dirname(log_path))
            with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
                for line in p.stdout:  # type: ignore[arg-type]
                    # echo to console for live per-prompt/run status
                    print(line, end="", flush=True)
                    logf.write(line)
            return p.wait()
    else:
        with subprocess.Popen(cmd, env=proc_env) as p:
            return p.wait()


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_index(index_rows: List[Dict[str, Any]], runs_dir: str) -> None:
    json_path = os.path.join(runs_dir, "index.json")
    csv_path = os.path.join(runs_dir, "index.csv")
    write_json(json_path, index_rows)
    if index_rows:
        fieldnames = list(index_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in index_rows:
                w.writerow(row)


def orchestrate(plan: Dict[str, Any], repo_root: str) -> int:
    prompts_file = plan.get("prompts_file", "prompts.json")
    # 'repeat' is no longer supported; always run the suite once per model
    repeat = 1
    pause_between_jobs_sec = int(plan.get("pause_between_jobs_sec", 30))
    retry = int(plan.get("retry", 0))
    resume = bool(plan.get("resume", True))
    # Always use results directory for all artifacts (evals, index). One folder per entry.
    results_root = os.path.abspath(plan.get("results_dir", os.path.join(repo_root, "results")))
    max_payload_chars = plan.get("judge", {}).get("max_payload_chars")

    bench_flags = plan.get("bench_flags", {})
    bench_exclude_warmup = bool(bench_flags.get("exclude_warmup_from_csv", True))
    bench_debug_usage = bool(bench_flags.get("debug_usage", False))
    bench_debug_network = bool(bench_flags.get("debug_network", False))
    # Universal runs-per-prompt for the entire plan (root takes precedence over bench_flags)
    bench_runs_per_prompt = plan.get("runs_per_prompt", bench_flags.get("runs_per_prompt"))

    judge = plan.get("judge", {})
    eval_enabled = bool(judge) and bool(judge.get("enabled", True))
    judge_base_url = judge.get("base_url") if eval_enabled else None
    judge_model = judge.get("model") if eval_enabled else None
    judge_api_key = judge.get("api_key") if eval_enabled else None
    judge_timeout = int(judge.get("timeout", 60)) if eval_enabled else 60
    judge_rate_limit = float(judge.get("rate_limit", 0.0)) if eval_enabled else 0.0
    judge_json_mode = bool(judge.get("json_mode", False)) if eval_enabled else False
    judge_max_tokens = judge.get("max_tokens") if eval_enabled else None
    judge_temperature = judge.get("temperature") if eval_enabled else None
    judge_top_p = judge.get("top_p") if eval_enabled else None
    judge_progress_every = judge.get("progress_every") if eval_enabled else None
    judge_verbose = bool(judge.get("verbose", False)) if eval_enabled else False
    judge_raw_column = bool(judge.get("raw_column", False)) if eval_enabled else False
    judge_dump_failures = judge.get("dump_failures") if eval_enabled else None

    matrix = plan.get("matrix", [])
    if not matrix:
        print("No matrix jobs in plan.")
        return 1

    ensure_dir(results_root)
    index_rows: List[Dict[str, Any]] = []

    for job_idx, job in enumerate(matrix):
        job_name = str(job.get("name", "job"))
        backend_name = str(job.get("backend_name", ""))
        base_url = str(job.get("base_url", "")).strip()
        # Normalize models list to handle common YAML mistakes (nested list or quoted list)
        raw_models = job.get("models", [])
        if isinstance(raw_models, str):
            parsed = None
            try:
                parsed = ast.literal_eval(raw_models)
            except Exception:
                pass
            if isinstance(parsed, list):
                models = [str(m) for m in parsed]
            else:
                # allow comma-separated string as a fallback
                models = [s.strip() for s in raw_models.split(",") if s.strip()]
        elif isinstance(raw_models, list) and len(raw_models) == 1 and isinstance(raw_models[0], list):
            models = [str(m) for m in raw_models[0]]
        else:
            models = [str(m) for m in (raw_models or [])]
        tokenizer_model_id = job.get("tokenizer_model_id")
        job_env = {str(k): str(v) for k, v in (job.get("env") or {}).items()}

        if not base_url or not models:
            print(f"Skip job '{job_name}': missing base_url or models")
            continue

        available_models = get_models(base_url, debug=bench_debug_network)

        for model_idx, model in enumerate(models):
            for rep in range(1, repeat + 1):
                ts = now_ts()
                # Entry folder: results/<timestamp>_<job>_<model>/
                entry_name = f"{ts}_{make_safe_segment(job_name)}_{make_safe_segment(model)}"
                out_dir = os.path.join(results_root, entry_name)
                ensure_dir(out_dir)

                meta = {
                    "name": job_name,
                    "model": model,
                    "backend": backend_name,
                    "base_url": base_url,
                    "prompts_file": prompts_file,
                    "rep": rep,
                    "ts": ts,
                }
                write_json(os.path.join(out_dir, "metadata.json"), meta)

                # Write bench CSV to a temp file; final artifact is only eval CSV
                bench_csv = os.path.join(results_root, f".tmp_bench_{entry_name}.csv")
                eval_csv = os.path.join(out_dir, f"{entry_name}_eval.csv")

                print(f"[job] {job_name} | model={model}")

                # Resume: if eval already exists, assume completed
                if resume and os.path.isfile(eval_csv):
                    index_rows.append({
                        "name": job_name,
                        "model": model,
                        "status": "skipped_resume",
                        "dir": out_dir,
                    })
                    print("[job] skipped (resume): eval.csv exists")
                    continue

                # Preflight: skip if /v1/models known and does not include model
                if available_models and model.lower() not in available_models:
                    print(f"[job] SKIP: model '{model}' not found in {base_url}")
                    index_rows.append({
                        "name": job_name,
                        "model": model,
                        "status": "skipped_missing_model",
                        "dir": out_dir,
                    })
                    continue

                # Use Python bench runner by default
                bench_cmd: List[str] = [
                    sys.executable, "-u", os.path.join(repo_root, "tools", "bench.py"),
                    "--base-url", base_url,
                    "--model", model,
                    "--backend-name", backend_name,
                    "--prompts-file", prompts_file,
                    "--out-csv", bench_csv,
                    "--timeout-sec", str(max(60, judge_timeout)),
                ]
                if tokenizer_model_id and tokenizer_model_id != "auto":
                    bench_cmd += ["--tokenizer-model-id", str(tokenizer_model_id)]
                if bench_exclude_warmup:
                    bench_cmd += ["--exclude-warmup-from-csv"]
                if bench_debug_usage:
                    bench_cmd += ["--debug-usage"]
                if bench_debug_network:
                    bench_cmd += ["--debug-network"]
                if bench_runs_per_prompt:
                    bench_cmd += ["--runs-per-prompt", str(bench_runs_per_prompt)]

                if eval_enabled:
                    eval_cmd: List[str] = [
                        sys.executable, "-u", os.path.join(repo_root, "tools", "evaluate.py"),
                        "--csv", bench_csv,
                        "--prompts", prompts_file,
                        "--out", eval_csv,
                    ]
                    if judge_base_url:
                        eval_cmd += ["--base-url", str(judge_base_url)]
                    if judge_model:
                        eval_cmd += ["--model", str(judge_model)]
                    if judge_api_key:
                        eval_cmd += ["--api-key", str(judge_api_key)]
                    if judge_timeout:
                        eval_cmd += ["--timeout", str(judge_timeout)]
                    if judge_rate_limit:
                        eval_cmd += ["--rate-limit", str(judge_rate_limit)]
                    if judge_json_mode:
                        eval_cmd += ["--json-mode"]
                    if judge_max_tokens:
                        eval_cmd += ["--max-tokens", str(judge_max_tokens)]
                    if judge_temperature is not None:
                        eval_cmd += ["--temperature", str(judge_temperature)]
                    if judge_top_p is not None:
                        eval_cmd += ["--top-p", str(judge_top_p)]
                    if judge_progress_every:
                        eval_cmd += ["--progress-every", str(judge_progress_every)]
                    else:
                        eval_cmd += ["--progress-every", "1"]
                    if judge_verbose:
                        eval_cmd += ["--verbose"]
                    if judge_raw_column:
                        eval_cmd += ["--raw-column"]
                    if judge_dump_failures:
                        eval_cmd += ["--dump-failures", str(judge_dump_failures)]
                    if max_payload_chars:
                        eval_cmd += ["--max-payload-chars", str(max_payload_chars)]

                # Run bench with optional retries
                attempts_left = 1 + max(0, retry)
                bench_rc = 1
                while attempts_left > 0:
                    attempts_left -= 1
                    print(f"[bench] start (attempt {1 + max(0, retry) - attempts_left}/{1 + max(0, retry)})")
                    bench_rc = run_process(bench_cmd, log_path=None, env=job_env)
                    if bench_rc == 0 and os.path.isfile(bench_csv):
                        print("[bench] ok")
                        break
                    if attempts_left > 0:
                        print(f"bench failed (rc={bench_rc}); retrying in 5s...")
                        time.sleep(5)

                status = "bench_failed" if (bench_rc != 0 or not os.path.isfile(bench_csv)) else "bench_ok"

                if status == "bench_ok":
                    if eval_enabled:
                        print("[eval] start")
                        eval_rc = run_process(eval_cmd, log_path=None, env=job_env)
                        print("[eval] ok" if (eval_rc == 0 and os.path.isfile(eval_csv)) else f"[eval] failed rc={eval_rc}")
                        status = "ok" if (eval_rc == 0 and os.path.isfile(eval_csv)) else "eval_failed"
                    else:
                        status = "ok_no_eval"

                index_rows.append({
                    "name": job_name,
                    "model": model,
                    "status": status,
                    "dir": out_dir,
                })

                # Cleanup temporary bench CSV if present
                try:
                    if os.path.isfile(bench_csv):
                        os.remove(bench_csv)
                except Exception:
                    pass

                # Cooldown between jobs
                if pause_between_jobs_sec > 0:
                    is_last = (job_idx == len(matrix) - 1 and model_idx == len(models) - 1)
                    if not is_last:
                        time.sleep(pause_between_jobs_sec)

    write_index(index_rows, results_root)
    print(f"Done. Wrote index under {results_root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch orchestrator for npuBench")
    parser.add_argument("--plan", default="tools/plan.yaml", help="Path to plan YAML/JSON")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    plan = read_plan(args.plan)
    return orchestrate(plan, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())


