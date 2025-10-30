#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import json
import os
import sys
import time
import glob
import subprocess
from datetime import datetime

import pandas as pd
import streamlit as st


WORKDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EVALUATE_SCRIPT = os.path.join(WORKDIR, 'tools', 'evaluate.py')
RESULTS_DIR = os.path.join(WORKDIR, 'results')


def find_csv_candidates():
    paths = []
    # Prefer results directory
    if os.path.isdir(RESULTS_DIR):
        paths.extend(sorted(glob.glob(os.path.join(RESULTS_DIR, '*.csv'))))
    # Fallback: search shallow under workdir
    if not paths:
        paths.extend(sorted(glob.glob(os.path.join(WORKDIR, '*.csv'))))
    return paths


def run_evaluator(
    csv_path: str,
    prompts_path: str,
    base_url: str,
    model: str,
    api_key: str | None,
    max_tokens: int | None,
    timeout_s: int,
    json_mode: bool,
    max_payload_chars: int,
    rate_limit: float,
    dump_failures_path: str | None,
    raw_column: bool,
    verbose: bool,
):
    cmd = [sys.executable, EVALUATE_SCRIPT,
           '--csv', csv_path,
           '--prompts', prompts_path,
           '--base-url', base_url,
           '--model', model,
           '--timeout', str(timeout_s)]

    if api_key:
        cmd += ['--api-key', api_key]
    if max_tokens is not None:
        cmd += ['--max-tokens', str(max_tokens)]
    if json_mode:
        cmd += ['--json-mode']
    if max_payload_chars and max_payload_chars > 0:
        cmd += ['--max-payload-chars', str(int(max_payload_chars))]
    if rate_limit and rate_limit > 0:
        cmd += ['--rate-limit', str(rate_limit)]
    if dump_failures_path:
        cmd += ['--dump-failures', dump_failures_path]
    if raw_column:
        cmd += ['--raw-column']
    if verbose:
        cmd += ['--verbose', '--progress-every', '1']

    st.code(' '.join(cmd), language='bash')
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKDIR)
    duration = time.time() - start
    return proc, duration


def load_csv_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except Exception:
        return pd.read_csv(path)


def read_prompts_any(path: str):
    try:
        txt = open(path, 'r', encoding='utf-8-sig').read()
    except Exception:
        txt = open(path, 'r').read()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and 'prompts' in obj:
            items = obj['prompts']
        elif isinstance(obj, list):
            items = obj
        else:
            items = [obj]
    except Exception:
        items = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    # Build id->item map
    pmap = {}
    idx = 0
    for it in items:
        idx += 1
        pid = it.get('id') if isinstance(it, dict) else None
        if not pid:
            pid = f'prompt_{idx}'
        pmap[pid] = it
    return pmap


def main():
    st.set_page_config(page_title='npuBench Evaluator UI', layout='wide')
    st.title('npuBench Evaluator')
    ss = st.session_state

    with st.sidebar:
        st.header('Inputs')
        # CSV selection
        csv_candidates = find_csv_candidates()
        csv_choice = st.selectbox('Benchmark CSV', options=['(upload)'] + csv_candidates, index=(0 if 'csv_choice' not in ss else (['(upload)'] + csv_candidates).index(ss['csv_choice']) if ss.get('csv_choice') in (['(upload)'] + csv_candidates) else 0))
        uploaded_csv = None
        if csv_choice == '(upload)':
            uploaded_csv = st.file_uploader('Upload CSV', type=['csv'])

        # Prompts file
        default_prompts = os.path.join(WORKDIR, 'prompts.json')
        prompts_path = st.text_input('Prompts JSON', value=default_prompts)
        st.session_state['prompts_path'] = prompts_path

        # Judge settings
        st.subheader('Judge server')
        base_url = st.text_input('Base URL', value='http://127.0.0.1:8000/v1/')
        model = st.text_input('Model', value='gpt-oss-20b-cuda-gpu:1')
        api_key = st.text_input('API Key (optional)', value='', type='password')

        # Controls
        st.subheader('Controls')
        max_tokens = st.number_input('max_tokens (optional)', value=256, min_value=0, step=64)
        timeout_s = st.number_input('HTTP timeout (s)', value=180, min_value=30, step=30)
        json_mode = st.checkbox('JSON-only mode', value=True, help='response_format=json_object')
        max_payload_chars = st.number_input('Clip payload to N chars (0=off)', value=4000, min_value=0, step=500)
        rate_limit = st.number_input('Rate limit between calls (seconds)', value=0.2, min_value=0.0, step=0.1, format='%0.1f')
        raw_column = st.checkbox('Include raw judge text in CSV', value=False)
        verbose = st.checkbox('Verbose evaluator logs', value=True)

        run_eval = st.button('Run evaluation')
        load_only = st.button('Load evaluated CSV')

    # Resolve CSV path
    csv_path = None
    temp_csv_path = None
    if uploaded_csv is not None:
        # Save to temp within results for consistency
        os.makedirs(RESULTS_DIR, exist_ok=True)
        temp_csv_path = os.path.join(RESULTS_DIR, f'uploaded_{int(time.time())}.csv')
        with open(temp_csv_path, 'wb') as f:
            f.write(uploaded_csv.read())
        csv_path = temp_csv_path
    elif csv_choice != '(upload)':
        csv_path = csv_choice
    # Persist CSV choice/path
    ss['csv_choice'] = csv_choice
    if csv_path:
        ss['csv_path'] = csv_path

    # Target failure log
    dump_failures_path = os.path.join(RESULTS_DIR, 'judge_failures.jsonl') if os.path.isdir(RESULTS_DIR) else 'judge_failures.jsonl'

    # Run evaluator if requested
    eval_output_path = ss.get('eval_output_path')
    eval_stdout = None
    eval_stderr = None
    eval_duration = None
    if run_eval:
        if not csv_path:
            st.error('Please select or upload a CSV.')
        elif not os.path.isfile(prompts_path):
            st.error('Prompts file not found.')
        else:
            with st.spinner('Running evaluator...'):
                proc, eval_duration = run_evaluator(
                    csv_path=csv_path,
                    prompts_path=prompts_path,
                    base_url=base_url,
                    model=model,
                    api_key=(api_key or None),
                    max_tokens=(int(max_tokens) if max_tokens > 0 else None),
                    timeout_s=int(timeout_s),
                    json_mode=bool(json_mode),
                    max_payload_chars=int(max_payload_chars),
                    rate_limit=float(rate_limit),
                    dump_failures_path=dump_failures_path,
                    raw_column=bool(raw_column),
                    verbose=bool(verbose),
                )
                eval_stdout = proc.stdout
                eval_stderr = proc.stderr
                # Try to locate output path from stdout last line
                out_path = None
                if eval_stdout:
                    for line in eval_stdout.splitlines()[::-1]:
                        if line.strip().lower().startswith('wrote:'):
                            out_path = line.split(':', 1)[1].strip()
                            break
                if out_path and os.path.isfile(out_path):
                    ss['eval_output_path'] = out_path
                    eval_output_path = out_path

    # Decide which CSV to show: evaluated if available, otherwise selected CSV
    display_path = ss.get('display_path')
    if eval_output_path and os.path.isfile(eval_output_path):
        display_path = eval_output_path
    elif load_only:
        # Prefer a previously parsed/evaluated CSV if exists
        if csv_path and csv_path.endswith('.csv'):
            base, ext = os.path.splitext(csv_path)
            candidate = f"{base}_eval.csv"
            if os.path.isfile(candidate):
                display_path = candidate
            else:
                display_path = csv_path
    # Persist chosen display path to survive widget-triggered reruns
    if display_path and os.path.isfile(display_path):
        ss['display_path'] = display_path

    # Logs panel
    with st.expander('Evaluator logs', expanded=False):
        cols = st.columns(3)
        with cols[0]:
            st.caption('Command output (stdout)')
            st.code(eval_stdout or '', language='bash')
        with cols[1]:
            st.caption('Command errors (stderr)')
            st.code(eval_stderr or '', language='bash')
        with cols[2]:
            st.caption('Judge failures (jsonl)')
            if os.path.isfile(dump_failures_path):
                try:
                    txt = open(dump_failures_path, 'r', encoding='utf-8').read()
                except Exception:
                    txt = open(dump_failures_path, 'r').read()
                st.code(txt, language='json')
            else:
                st.write('(none)')

    # Visualization
    if display_path and os.path.isfile(display_path):
        st.success(f'Displaying: {display_path}')
        df = load_csv_any(display_path)

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        total_rows = len(df)
        ok_rows = (df['eval_status'] == 'ok').sum() if 'eval_status' in df.columns else 0
        parse_fail = (df['eval_status'] == 'judge_parse_failed').sum() if 'eval_status' in df.columns else 0
        errors = df['eval_status'].astype(str).str.startswith('judge_error').sum() if 'eval_status' in df.columns else 0
        skipped = (df['eval_status'] == 'skipped_duplicate_output').sum() if 'eval_status' in df.columns else 0
        with c1: st.metric('Total rows', total_rows)
        with c2: st.metric('OK', int(ok_rows))
        with c3: st.metric('Parse fails', int(parse_fail))
        with c4: st.metric('Errors', int(errors))
        with c5: st.metric('Skipped dupes', int(skipped))

        # Verdict counts
        if 'eval_verdict' in df.columns:
            verdict_counts = df['eval_verdict'].fillna('(none)').value_counts().reset_index()
            verdict_counts.columns = ['eval_verdict', 'count']
            st.bar_chart(verdict_counts.set_index('eval_verdict'))

        # Score distribution (OK rows only)
        if 'eval_score_0_5' in df.columns and 'eval_status' in df.columns:
            st.subheader('Scores (OK only)')
            ok_scores = df[df['eval_status'] == 'ok']['eval_score_0_5'].dropna()
            try:
                ok_scores = ok_scores.astype(float)
            except Exception:
                pass
            if not ok_scores.empty:
                st.bar_chart(ok_scores.value_counts().sort_index())
            else:
                st.info('No OK rows with scores to plot.')

        # Rows table and Row inspector removed per request

        # Compare view: unique completed evaluations (skip skipped duplicates)
        st.subheader('Compare: Candidate vs Reference (unique completed)')
        if 'eval_status' in df.columns and 'prompt_id' in df.columns:
            df_completed = df[df['eval_status'] == 'ok'].copy()
            df_completed = df_completed[df_completed['prompt_id'].notna()]
            # Keep first completed per prompt_id
            df_unique = df_completed.sort_values(['prompt_id', 'run_index']).drop_duplicates(subset=['prompt_id'], keep='first')
            if df_unique.empty:
                st.info('No completed evaluations to display yet.')
            else:
                # Load references
                prompts_path_resolved = st.session_state.get('prompts_path') or os.path.join(WORKDIR, 'prompts.json')
                pmap = read_prompts_any(prompts_path_resolved) if os.path.isfile(prompts_path_resolved) else {}

                # Build selection
                pid_options = list(df_unique['prompt_id'].astype(str).values)
                default_pid = ss.get('compare_pid', pid_options[0]) if pid_options else None
                if default_pid not in pid_options and pid_options:
                    default_pid = pid_options[0]
                sel_pid = st.selectbox('Select prompt', options=pid_options, index=(pid_options.index(default_pid) if default_pid in pid_options else 0))
                ss['compare_pid'] = sel_pid

                row_u = df_unique[df_unique['prompt_id'].astype(str) == str(sel_pid)].iloc[0]
                ref_item = pmap.get(sel_pid) or {}
                reference_text = ref_item.get('reference') or ''

                c1, c2 = st.columns(2)
                with c1:
                    st.caption('Candidate (completion_text)')
                    st.text_area('cmp_candidate', value=row_u.get('completion_text') or '', height=360, label_visibility='collapsed')
                with c2:
                    st.caption('Reference (from prompts.json)')
                    st.text_area('cmp_reference', value=reference_text, height=240, label_visibility='collapsed')

                # Show only rationale and raw judge response
                st.caption('Rationale')
                st.text_area('cmp_rationale', value=row_u.get('eval_rationale') or '', height=120, label_visibility='collapsed')
                if 'eval_raw_judge' in row_u:
                    st.caption('Raw judge response')
                    st.text_area('cmp_raw_judge', value=row_u.get('eval_raw_judge') or '', height=120, label_visibility='collapsed')


if __name__ == '__main__':
    main()


