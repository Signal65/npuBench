#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time

import pandas as pd
import streamlit as st


WORKDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(WORKDIR, 'results')


def load_csv_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except Exception:
        return pd.read_csv(path)


def find_eval_csv_candidates():
    paths = []
    if os.path.isdir(RESULTS_DIR):
        paths.extend(sorted(glob.glob(os.path.join(RESULTS_DIR, '*_eval.csv'))))
    if not paths:
        paths.extend(sorted(glob.glob(os.path.join(WORKDIR, '*_eval.csv'))))
    return paths


def safe_mean(series):
    try:
        return pd.to_numeric(series, errors='coerce').dropna().mean()
    except Exception:
        return float('nan')


def main():
    st.set_page_config(page_title='npuBench Dashboard', layout='wide')
    # Top navigation
    nav = st.columns([1,1,8])
    try:
        with nav[0]:
            st.page_link('app.py', label='Evaluator', icon='🧪')
        with nav[1]:
            st.page_link('pages/1_Dashboard.py', label='Dashboard', icon='📊')
    except Exception:
        with nav[0]:
            st.write('🧪 Evaluator (use sidebar pages)')
        with nav[1]:
            st.write('📊 Dashboard')
    st.title('Dashboard: Compare systems/models/backends')

    d_cols = st.columns([3, 1])
    with d_cols[0]:
        eval_candidates = find_eval_csv_candidates()
        picked = st.multiselect('Evaluated CSVs', options=eval_candidates, default=eval_candidates[:2])
    with d_cols[1]:
        add_files = st.file_uploader('Add evaluated CSVs', type=['csv'], accept_multiple_files=True)

    all_paths = list(picked)
    if add_files:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        for fobj in add_files:
            tmp_path = os.path.join(RESULTS_DIR, f"dash_{int(time.time())}_{os.path.basename(fobj.name)}")
            with open(tmp_path, 'wb') as fp:
                fp.write(fobj.read())
            all_paths.append(tmp_path)

    if not all_paths:
        st.info('Pick one or more evaluated CSVs to compare.')
        return

    frames = []
    for pth in all_paths:
        try:
            dfp = load_csv_any(pth)
            dfp['source_file'] = pth
            frames.append(dfp)
        except Exception as e:
            st.warning(f'Failed to load {pth}: {e}')

    if not frames:
        st.warning('No valid CSVs loaded.')
        return

    dfa = pd.concat(frames, ignore_index=True)

    # Summary by file (accuracy + performance)
    summary = dfa.groupby('source_file').apply(lambda g: pd.Series({
        'model': g.get('model', pd.Series([None])).iloc[0] if 'model' in g.columns else '',
        'backend': g.get('backend_name', pd.Series([None])).iloc[0] if 'backend_name' in g.columns else '',
        'rows': len(g),
        'ok': int((g['eval_status'] == 'ok').sum()) if 'eval_status' in g.columns else 0,
        'parse_failed': int((g['eval_status'] == 'judge_parse_failed').sum()) if 'eval_status' in g.columns else 0,
        'errors': int(g['eval_status'].astype(str).str.startswith('judge_error').sum()) if 'eval_status' in g.columns else 0,
        'skipped_dupe': int((g['eval_status'] == 'skipped_duplicate_output').sum()) if 'eval_status' in g.columns else 0,
        'avg_score_ok': safe_mean(g.loc[g.get('eval_status','')=='ok','eval_score_0_5']) if 'eval_score_0_5' in g.columns else float('nan'),
        'avg_ttft_ms': safe_mean(g['ttft_ms']) if 'ttft_ms' in g.columns else float('nan'),
        'avg_gen_tps': safe_mean(g['gen_tokens_per_s']) if 'gen_tokens_per_s' in g.columns else float('nan'),
    })).reset_index()

    st.subheader('Summary by file')
    st.dataframe(summary, use_container_width=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.caption('Average score (OK only)')
        if not summary.empty:
            st.bar_chart(data=summary.set_index('source_file')['avg_score_ok'])
    with c2:
        st.caption('Average TTFT (ms)')
        if not summary.empty:
            st.bar_chart(data=summary.set_index('source_file')['avg_ttft_ms'])

    # Per-prompt comparison across files
    st.subheader('Per-prompt comparison across files')
    common_pids = sorted(set(dfa.get('prompt_id', pd.Series(dtype=str)).dropna().unique().tolist()))
    if common_pids:
        sel_pid_dash = st.selectbox('Prompt', options=common_pids)
        rows = dfa[dfa['prompt_id'].astype(str) == str(sel_pid_dash)].copy()
        # keep one representative per file (prefer OK; else first by run_index)
        def pick_row(g):
            if 'eval_status' in g.columns:
                ok = g[g['eval_status']=='ok']
                if not ok.empty:
                    return ok.sort_values('run_index').iloc[0]
            return g.sort_values('run_index').iloc[0]
        picked_rows = rows.groupby('source_file').apply(pick_row).reset_index(drop=True)
        view = picked_rows[['source_file','model','backend_name','eval_score_0_5','eval_verdict','ttft_ms','gen_tokens_per_s','completion_text']].copy()
        # Avoid pyarrow json serialization issues by stringifying non-numerics
        for col in ['source_file','model','backend_name','eval_verdict','completion_text']:
            if col in view.columns:
                view[col] = view[col].astype(str)
        st.dataframe(view, use_container_width=True)


if __name__ == '__main__':
    main()


