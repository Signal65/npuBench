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

    # Folder-based bulk selection (auto-load)
    st.subheader('Select results')
    fc1, fc2 = st.columns([4, 1])
    with fc1:
        folder_path = st.text_input('Results folder', value=RESULTS_DIR)
    with fc2:
        recurse = st.checkbox('Recurse', value=True)

    # Normalize and expand the provided path
    folder_norm = os.path.normpath(os.path.expanduser(os.path.expandvars(folder_path.strip())))

    # Manual selection remains available (optional)
    d_cols = st.columns([3, 1])
    with d_cols[0]:
        eval_candidates = find_eval_csv_candidates()
        picked = st.multiselect('Or pick specific evaluated CSVs', options=eval_candidates, default=[])
    with d_cols[1]:
        add_files = st.file_uploader('Or add files…', type=['csv'], accept_multiple_files=True)

    # Auto-load from folder (supports both *_eval.csv and legacy eval.csv)
    all_paths = []
    if os.path.isdir(folder_norm):
        patterns = []
        if recurse:
            patterns = [
                os.path.join(folder_norm, '**', '*_eval.csv'),
                os.path.join(folder_norm, '**', 'eval.csv'),
            ]
        else:
            patterns = [
                os.path.join(folder_norm, '*_eval.csv'),
                os.path.join(folder_norm, 'eval.csv'),
            ]
        found = []
        for pat in patterns:
            found.extend(glob.glob(pat, recursive=recurse))
        # Deduplicate while preserving order
        seen = set()
        for p in found:
            if p not in seen:
                seen.add(p)
                all_paths.append(p)
        st.caption(f"Loaded {len(all_paths)} file(s) from folder: {folder_norm}")
    else:
        st.warning('Folder not found. Please check the path.')

    if picked:
        all_paths.extend(list(picked))
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

    # Summary by file using requested fields; averages exclude warmup runs
    def summarize_file(g: pd.DataFrame) -> pd.Series:
        # Warmup mask (treat string booleans)
        warm = g.get('is_warmup')
        is_warm = pd.Series([False] * len(g))
        if warm is not None:
            is_warm = warm.astype(str).str.lower().isin(['true', '1', 'yes'])
        # Duplicate mask
        dup = g.get('is_duplicate')
        is_dup = pd.Series([False] * len(g))
        if dup is not None:
            is_dup = dup.astype(str).str.lower().isin(['true', '1', 'yes'])
        # Filter out warmups for latency/token rates and filter out duplicates for all aggregates
        g_nowarm = g[(~is_warm) & (~is_dup)]
        # OK rows for scoring
        ok_mask = (g.get('eval_status', pd.Series([])).astype(str) == 'ok') if 'eval_status' in g.columns else pd.Series([True]*len(g))
        g_ok = g[ok_mask & (~is_dup)]
        # Non-duplicate rows for totals
        total_tests = int((~is_dup).sum())
        # Passed = non-duplicate rows graded as correct or partial
        verdict_series = g.get('eval_verdict')
        if verdict_series is not None:
            verdict_ok = verdict_series.astype(str).str.lower().isin(['correct', 'partial'])
        else:
            verdict_ok = pd.Series([False] * len(g))
        passed_tests = int((~is_dup & verdict_ok).sum())
        return pd.Series({
            'timestamp': (g['timestamp'].dropna().astype(str).iloc[0] if 'timestamp' in g.columns and not g['timestamp'].dropna().empty else ''),
            'system_model': (g['system_model'].dropna().astype(str).iloc[0] if 'system_model' in g.columns and not g['system_model'].dropna().empty else ''),
            'processor_name': (g['processor_name'].dropna().astype(str).iloc[0] if 'processor_name' in g.columns and not g['processor_name'].dropna().empty else ''),
            'ttft_ms': safe_mean(g_nowarm['ttft_ms']) if 'ttft_ms' in g_nowarm.columns else float('nan'),
            'gen_tokens_per_s': safe_mean(g_nowarm['gen_tokens_per_s']) if 'gen_tokens_per_s' in g_nowarm.columns else float('nan'),
            'eval_score': safe_mean(g_ok['eval_score_0_5']) if 'eval_score_0_5' in g_ok.columns else float('nan'),
            'passed_total': f'{passed_tests}/{total_tests}',
        })
    summary = dfa.groupby('source_file').apply(summarize_file).reset_index()

    st.subheader('Summary by file')
    # Stringify non-numerics for safe rendering
    for col in ['timestamp','system_model','processor_name']:
        if col in summary.columns:
            summary[col] = summary[col].astype(str)
    st.dataframe(summary[['source_file','timestamp','system_model','processor_name','ttft_ms','gen_tokens_per_s','eval_score','passed_total']], use_container_width=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.caption('Average score (OK only)')
        if not summary.empty:
            st.bar_chart(data=summary.set_index('source_file')['eval_score'])
    with c2:
        st.caption('Average TTFT (ms, no warmups)')
        if not summary.empty:
            st.bar_chart(data=summary.set_index('source_file')['ttft_ms'])

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
        # Prepare requested fields
        desired = ['timestamp','system_model','processor_name','ttft_ms','gen_tokens_per_s','eval_score_0_5']
        for col in desired:
            if col not in picked_rows.columns:
                picked_rows[col] = ''
        view = picked_rows[desired].copy()
        # Rename for display
        view = view.rename(columns={'eval_score_0_5': 'eval_score'})
        # Stringify non-numerics for safe rendering
        for col in ['timestamp','system_model','processor_name']:
            if col in view.columns:
                view[col] = view[col].astype(str)
        st.dataframe(view, use_container_width=True)


if __name__ == '__main__':
    main()


