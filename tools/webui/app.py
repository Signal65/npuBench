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
    if os.path.isdir(RESULTS_DIR):
        paths.extend(sorted(glob.glob(os.path.join(RESULTS_DIR, '*.csv'))))
    if not paths:
        paths.extend(sorted(glob.glob(os.path.join(WORKDIR, '*.csv'))))
    return paths


def find_eval_csv_candidates():
    paths = []
    if os.path.isdir(RESULTS_DIR):
        paths.extend(sorted(glob.glob(os.path.join(RESULTS_DIR, '*_eval.csv'))))
    if not paths:
        paths.extend(sorted(glob.glob(os.path.join(WORKDIR, '*_eval.csv'))))
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
    st.set_page_config(page_title='Home', layout='wide')
    st.title('Home')
    st.subheader('Choose a view')
    try:
        c1, c2 = st.columns(2)
        with c1:
            if st.button('🧪 Evaluator', use_container_width=True):
                st.switch_page('pages/2_Evaluator.py')  # type: ignore[attr-defined]
        with c2:
            if st.button('📊 Dashboard', use_container_width=True):
                st.switch_page('pages/1_Dashboard.py')  # type: ignore[attr-defined]
    except Exception:
        st.info('Use the sidebar Pages to open Evaluator or Dashboard.')
    return


if __name__ == '__main__':
    main()
