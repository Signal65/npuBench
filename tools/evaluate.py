#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Offline evaluator for npuBench CSVs using an OpenAI-compatible judge.

Usage:
  python tools/evaluate.py \
    --csv results/bench_YYYYMMDD_HHMMSS.csv \
    --prompts prompts.json \
    --base-url http://127.0.0.1:8000/v1/ \
    --model gpt-judge-model \
    [--api-key YOUR_KEY] [--out results_eval.csv]

Notes:
- Works with any OpenAI-compatible /v1/chat/completions endpoint.
- Temperature fixed to 0 by default for determinism.
- Prompts can optionally include fields per item:
  - reference: string reference answer
  - reference_regex: regex string for exact/regex pre-match
  - numeric_answer: number for numeric comparison
  - task_type: optional tag (openorca_qa, summarization, etc.)
- Rows lacking a reference are marked with eval_status=no_reference and skipped.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from urllib import request, error


def read_json_any(path: str):
    text = open(path, 'r', encoding='utf-8-sig').read()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'prompts' in obj and isinstance(obj['prompts'], list):
            return obj['prompts']
        if isinstance(obj, list):
            return obj
        return [obj]
    except Exception:
        # JSONL
        items = []
        for line in open(path, 'r', encoding='utf-8-sig'):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
        return items


def build_prompt_map(items):
    m = {}
    idx = 0
    for it in items:
        idx += 1
        pid = it.get('id') if isinstance(it, dict) else None
        if not pid:
            pid = f"prompt_{idx}"
        m[pid] = it
    return m


def normalize_text(s: str) -> str:
    s = s or ''
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def try_numeric(s: str):
    try:
        return float(str(s).strip())
    except Exception:
        return None
def normalize_for_dup(s: str) -> str:
    s = s or ''
    return re.sub(r"\s+", " ", s).strip()



def prefilter_score(candidate: str, meta: dict):
    """Return (handled: bool, score: int, verdict: str, rationale: str)."""
    cand_norm = normalize_text(candidate)
    ref = (meta.get('reference') or '') if isinstance(meta, dict) else ''
    ref_norm = normalize_text(ref)

    # Empty candidate
    if not cand_norm:
        return True, 0, 'incorrect', 'Empty output'

    # Exact/normalized match
    if ref_norm and cand_norm == ref_norm:
        return True, 5, 'correct', 'Exact/normalized match'

    # Regex
    ref_re = meta.get('reference_regex') if isinstance(meta, dict) else None
    if ref_re:
        try:
            if re.search(ref_re, candidate, re.IGNORECASE | re.DOTALL):
                return True, 5, 'correct', 'Regex match'
        except re.error:
            pass

    # Numeric
    num_ans = meta.get('numeric_answer') if isinstance(meta, dict) else None
    if num_ans is not None:
        try:
            cnum = float(candidate)
            ans = float(num_ans)
            if math.isclose(cnum, ans, rel_tol=1e-3, abs_tol=1e-6):
                return True, 5, 'correct', 'Numeric close match'
            else:
                return True, 0, 'incorrect', f'Numeric mismatch: {cnum} vs {ans}'
        except Exception:
            return True, 0, 'incorrect', 'Numeric parse failed'

    return False, 0, 'unknown', ''


JUDGE_SYSTEM = (
    "You are a strict evaluator. Judge the candidate against the reference for factual correctness and directness. "
    "Respond with EXACTLY one JSON object and NOTHING else. "
    "Schema: {\"score\":0-5,\"verdict\":\"correct|partial|incorrect\",\"hallucination\":true|false,\"rationale\":\"1-3 sentences explaining the decision\"}. "
    "Rules: score is an integer 0..5; rationale should explicitly state why the answer is correct/partial/incorrect; no prose outside JSON."
)


def call_chat(base_url: str, api_key: str | None, model: str, messages: list, max_tokens: int | None = 128,
              temperature: float = 0.0, top_p: float = 1.0, timeout: int = 60,
              response_format_json: bool = False) -> str:
    url = base_url.rstrip('/') + '/chat/completions'
    body = {
        'model': model,
        'messages': messages,
        'temperature': float(temperature),
        'top_p': float(top_p),
        'stream': False,
    }
    if max_tokens is not None:
        body['max_tokens'] = int(max_tokens)
    if response_format_json:
        body['response_format'] = { 'type': 'json_object' }
    data = json.dumps(body).encode('utf-8')
    req = request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    if api_key:
        req.add_header('Authorization', f'Bearer {api_key}')
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode('utf-8', errors='replace')
            obj = json.loads(text)
            # Robust extraction across providers
            choices = obj.get('choices') or []
            if not choices:
                return ''
            choice = choices[0]
            msg = choice.get('message') or {}
            content = msg.get('content')
            # Some providers return list of parts for content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        parts.append(part.get('text', ''))
                    elif isinstance(part, str):
                        parts.append(part)
                content = '\n'.join([p for p in parts if p])
            if not content:
                # Legacy field
                content = choice.get('text') or ''
            return content or ''
    except error.HTTPError as e:
        txt = e.read().decode('utf-8', errors='replace')
        # Some servers require max_tokens. If omitted (max_tokens is None) and server complains, retry once with 2048.
        if (max_tokens is None) and ('max_tokens' in txt.lower() or 'max tokens' in txt.lower()):
            body_retry = dict(body)
            body_retry['max_tokens'] = 2048
            data_retry = json.dumps(body_retry).encode('utf-8')
            req_retry = request.Request(url, data=data_retry, headers={'Content-Type': 'application/json'})
            if api_key:
                req_retry.add_header('Authorization', f'Bearer {api_key}')
            with request.urlopen(req_retry, timeout=timeout) as resp2:
                text2 = resp2.read().decode('utf-8', errors='replace')
                obj2 = json.loads(text2)
                choice2 = (obj2.get('choices') or [{}])[0]
                msg2 = choice2.get('message') or {}
                return msg2.get('content') or ''
        raise RuntimeError(f'HTTP {e.code}: {txt}')


def extract_json(text: str):
    # Try whole response first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: search all brace blocks and return the LAST valid JSON object
    # This helps when the model mentions an example schema early and the real JSON appears later.
    candidates = list(re.finditer(r"\{[\s\S]*?\}", text))
    for m in reversed(candidates):
        frag = m.group(0)
        try:
            return json.loads(frag)
        except Exception:
            continue
    return None


essentials = [
    'timestamp','hostname','backend_name','base_url','model','prompt_id','run_index',
    'ttft_ms','prompt_tokens','completion_tokens','prompt_tokens_per_s','gen_tokens_per_s','completion_text'
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Input benchmark CSV')
    ap.add_argument('--prompts', required=True, help='Prompts file (JSON/JSONL)')
    ap.add_argument('--base-url', required=True, help='OpenAI-compatible base URL, e.g. http://host:port/v1/')
    ap.add_argument('--model', required=True, help='Judge model id')
    ap.add_argument('--api-key', default=None)
    ap.add_argument('--out', default=None, help='Output CSV (default: alongside input with _eval suffix)')
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--rate-limit', type=float, default=0.0, help='Seconds to sleep between judge calls')
    ap.add_argument('--verbose', action='store_true', help='Print progress and decisions')
    ap.add_argument('--progress-every', type=int, default=25, help='Print progress every N rows (0=off)')
    ap.add_argument('--dump-failures', default=None, help='Write JSONL with rows where judge parsing/errors occurred')
    ap.add_argument('--raw-column', action='store_true', help='Include eval_raw_judge column with raw judge text')
    ap.add_argument('--timeout', type=int, default=180, help='HTTP timeout (seconds) for judge requests')
    ap.add_argument('--json-mode', action='store_true', help='Set response_format=json_object for JSON-only responses (if server supports)')
    ap.add_argument('--max-payload-chars', type=int, default=0, help='If >0, clip candidate/reference to this many chars before judging')
    args = ap.parse_args()

    prompts = read_json_any(args.prompts)
    pmap = build_prompt_map(prompts)
    if args.verbose:
        print(f"Judge base-url: {args.base_url}")
        print(f"Judge model:    {args.model}")
        print(f"Prompts loaded: {len(pmap)} from {args.prompts}")

    in_rows = []
    with open(args.csv, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            in_rows.append(row)

    # Detect candidate text column from common options
    candidate_candidates = ['completion_text', 'output', 'text', 'response', 'answer']
    header_keys = list(in_rows[0].keys()) if in_rows else []
    candidate_col = next((c for c in candidate_candidates if c in header_keys), 'completion_text')
    if args.verbose:
        print(f"Using candidate column: {candidate_col}")

    # Detect groups where all runs have identical non-empty outputs; we'll evaluate only the first and skip the rest
    prompt_to_norm_outputs = {}
    for row in in_rows:
        pid0 = row.get('prompt_id')
        norm_out = normalize_for_dup(row.get(candidate_col) or '')
        if pid0 not in prompt_to_norm_outputs:
            prompt_to_norm_outputs[pid0] = set()
        prompt_to_norm_outputs[pid0].add(norm_out)

    duplicate_prompt_ids = set()
    for pid0, outs in prompt_to_norm_outputs.items():
        if len(outs) == 1:
            only = next(iter(outs))
            if only != '':
                duplicate_prompt_ids.add(pid0)

    out_rows = []
    total = len(in_rows)
    cnt_no_ref = 0
    cnt_prefilter = 0
    cnt_judged = 0
    cnt_judge_error = 0
    cnt_parse_fail = 0
    cnt_skipped_dupe = 0
    dupe_first_seen = set()
    failures_fp = None
    if args.dump_failures:
        failures_dir = os.path.dirname(args.dump_failures)
        if failures_dir and not os.path.isdir(failures_dir):
            os.makedirs(failures_dir, exist_ok=True)
        failures_fp = open(args.dump_failures, 'w', encoding='utf-8')
    if args.verbose:
        print(f"CSV rows:       {total} from {args.csv}")
    for idx, row in enumerate(in_rows, start=1):
        pid = row.get('prompt_id')
        cand = row.get(candidate_col) or ''
        meta = pmap.get(pid) or {}
        reference = meta.get('reference') if isinstance(meta, dict) else None

        eval_status = 'ok'
        score = ''
        verdict = ''
        rationale = ''
        hallucination = ''

        raw_judge_text = ''
        # Skip duplicate outputs for this prompt after the first occurrence
        if pid in duplicate_prompt_ids:
            if pid in dupe_first_seen:
                new_row = dict(row)
                new_row.update({
                    'eval_status': 'skipped_duplicate_output',
                    'eval_score_0_5': '',
                    'eval_verdict': '',
                    'eval_hallucination': '',
                    'eval_rationale': '',
                    'eval_model': args.model,
                    'is_duplicate': True,
                })
                if args.raw_column:
                    new_row['eval_raw_judge'] = ''
                out_rows.append(new_row)
                cnt_skipped_dupe += 1
                if args.verbose and args.progress_every and (idx % args.progress_every == 0 or idx == total):
                    print(f"progress {idx}/{total} | no_ref={cnt_no_ref} prefilter={cnt_prefilter} judged={cnt_judged} parse_fail={cnt_parse_fail} errors={cnt_judge_error} skipped={cnt_skipped_dupe}")
                continue
            else:
                dupe_first_seen.add(pid)

        if not reference:
            eval_status = 'no_reference'
            cnt_no_ref += 1
        else:
            handled, s, v, r = prefilter_score(cand, meta)
            if handled:
                score, verdict, rationale = s, v, r
                cnt_prefilter += 1
            else:
                # Optionally clip large payload fields to reduce latency/timeouts
                clipped_candidate = cand
                clipped_reference = reference
                if args.max_payload_chars and args.max_payload_chars > 0:
                    limit = int(args.max_payload_chars)
                    if len(clipped_candidate) > limit:
                        clipped_candidate = clipped_candidate[:limit]
                    if isinstance(clipped_reference, str) and len(clipped_reference) > limit:
                        clipped_reference = clipped_reference[:limit]

                user_payload = {
                    'prompt_id': pid,
                    'task_type': meta.get('task_type') if isinstance(meta, dict) else None,
                    'question_messages': meta.get('messages') if isinstance(meta, dict) else None,
                    'reference': clipped_reference,
                    'candidate': clipped_candidate,
                }
                messages = [
                    { 'role': 'system', 'content': JUDGE_SYSTEM },
                    { 'role': 'user', 'content': json.dumps(user_payload, ensure_ascii=False) }
                ]
                try:
                    content = call_chat(
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model,
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        timeout=args.timeout,
                        response_format_json=args.json_mode,
                    )
                    raw_judge_text = content or ''
                    obj = extract_json(content) or {}
                    # Fallback: if empty or unparsable, retry once without JSON mode
                    if (not obj) or (obj.get('score') is None and obj.get('verdict') is None):
                        if args.verbose:
                            prev = (raw_judge_text[:160] + '...') if len(raw_judge_text) > 160 else raw_judge_text
                            print(f"retry_no_json row={idx} pid={pid} preview={prev!r}")
                        content2 = call_chat(
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            messages=messages,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            timeout=args.timeout,
                            response_format_json=False,
                        )
                        if content2:
                            raw_judge_text = content2
                            obj = extract_json(content2) or {}
                    score = obj.get('score')
                    verdict = obj.get('verdict')
                    rationale = obj.get('rationale')
                    hallucination = obj.get('hallucination')
                    if score is None or verdict is None:
                        eval_status = 'judge_parse_failed'
                        cnt_parse_fail += 1
                        if failures_fp is not None:
                            failures_fp.write(json.dumps({
                                'row_index': idx,
                                'prompt_id': pid,
                                'eval_status': eval_status,
                                'user_payload': user_payload,
                                'raw_judge_text': raw_judge_text,
                            }, ensure_ascii=False) + "\n")
                        if args.verbose:
                            preview = (raw_judge_text[:160] + '...') if len(raw_judge_text) > 160 else raw_judge_text
                            print(f"parse_failed row={idx} pid={pid} preview={preview!r}")
                    else:
                        cnt_judged += 1
                except Exception as e:
                    eval_status = f'judge_error: {e}'
                    cnt_judge_error += 1
                    if failures_fp is not None:
                        failures_fp.write(json.dumps({
                            'row_index': idx,
                            'prompt_id': pid,
                            'eval_status': eval_status,
                            'user_payload': user_payload,
                        }, ensure_ascii=False) + "\n")
                    if args.verbose:
                        print(f"judge_error row={idx} pid={pid} err={e}")
                if args.rate_limit > 0:
                    time.sleep(args.rate_limit)

        if args.verbose and args.progress_every and (idx % args.progress_every == 0 or idx == total):
            print(f"progress {idx}/{total} | no_ref={cnt_no_ref} prefilter={cnt_prefilter} judged={cnt_judged} parse_fail={cnt_parse_fail} errors={cnt_judge_error} skipped={cnt_skipped_dupe}")

        new_row = dict(row)
        new_row.update({
            'eval_status': eval_status,
            'eval_score_0_5': score,
            'eval_verdict': verdict,
            'eval_hallucination': hallucination,
            'eval_rationale': rationale,
            'eval_model': args.model,
            'is_duplicate': False,
        })
        if args.raw_column:
            new_row['eval_raw_judge'] = raw_judge_text
        out_rows.append(new_row)

    out_path = args.out
    if not out_path:
        base, ext = os.path.splitext(args.csv)
        out_path = f"{base}_eval.csv"

    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        # keep original header order + eval fields at end
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote: {out_path}")
    if args.verbose:
        print(f"Summary: no_ref={cnt_no_ref}, prefilter={cnt_prefilter}, judged={cnt_judged}, parse_fail={cnt_parse_fail}, errors={cnt_judge_error}, skipped={cnt_skipped_dupe}")
    if failures_fp is not None:
        failures_fp.close()


if __name__ == '__main__':
    sys.exit(main())
