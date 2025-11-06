## npuBench - Windows NPU LLM Benchmark (PowerShell)

A lightweight PowerShell framework to benchmark OpenAI-compatible LLM backends (Nexa, FastFlowLM, etc.) running on NPU-enabled Windows machines.

- Backend-agnostic: targets any OpenAI-compatible endpoint
- Measures: time to first token (TTFT), prompt tokens/sec, generation tokens/sec
- Captures device info: OS, CPU, GPU(s), NPU(s), RAM
- Outputs a single CSV per run with metrics and context

### Requirements
- Windows PowerShell 5.1 or PowerShell 7+ (for the original script)
- Python 3.10+ (for the Python bench/orchestrator/web UI)
- Network access to an OpenAI-compatible server

### Python packages (test system)
Install these with pip (recommend a virtual environment):

Core (bench/orchestrator/evaluator):
```
python -m pip install requests pyyaml pandas
```

Web UI (Streamlit dashboard and evaluator UI):
```
python -m pip install streamlit pandas plotly pyarrow
```

Optional components:
- Tokenizer fallback for gen_tps when servers omit usage:
```
python -m pip install transformers tokenizers
```
- Parquet reference extractor (OpenOrca):
```
python -m pip install pandas pyarrow fastparquet
```

You can also install the web UI dependencies via:
```
python -m pip install -r tools/webui/requirements.txt
```

Notes:
- PyTorch/TF/Flax are NOT required for token counting; tokenizers work without them.
- If running offline, cache tokenizers first or use `--force-tokenizer` after caching.

### Usage
Run from this directory:

```powershell
# Example using host/port
./bench.ps1 -ServerHost 192.168.1.50 -Port 8000 -Model "qwen2.5-7b-instruct" -BackendName "FastFlowLM"

# Example using a full base URL
./bench.ps1 -BaseUrl "http://192.168.1.50:8000/v1" -Model "llama-3.1-8b-instruct" -BackendName "Nexa"

# Options
./bench.ps1 -ServerHost <ip> -Port <port> -Model <modelName> -BackendName <name> -RunsPerPrompt 4 -MaxOutputTokens <int?> -Temperature 0 -TopP 1 -Seed 0 -OutputDir ./results -TokenizerModelId <hf_tokenizer_id?> -PythonPath python -TokenizerLocalOnly -DebugTokenizer
```

Python bench (no PowerShell required):
```
python tools/bench.py --base-url http://192.168.1.50:8000/v1 \
  --model llama-3.1-8b-instruct --backend-name Nexa \
  --prompts-file prompts.json --out-csv results.csv --runs-per-prompt 4
```

Python orchestrator (matrix of models/backends):
```
python tools/orchestrate.py --plan tools/plan.yaml
```

Web UI (evaluator and comparison dashboard):
```
streamlit run tools/webui/app.py
```

Notes:
- If your server uses HTTPS with a self-signed certificate, add `-SkipCertificateCheck`.
- The script attempts to obtain token usage from the server. If usage is not sent in stream, it falls back to either a non-stream request for usage, or (optionally) local tokenization via Hugging Face.

### Output
CSV written to `results/` with columns including:
- timestamp, hostname, os_version, cpu_name, gpu_names, npu_names, ram_gb
- backend_name, base_url, model, prompt_id, run_index
- ttft_ms, prompt_tokens, prompt_tokens_per_s
- completion_tokens, gen_tokens_per_s, total_time_ms
- temperature, top_p, seed, runs_per_prompt, max_output_tokens

### Extending
- PowerShell path: baked prompts under `$BenchPrompts` in `bench.ps1`; HTTP logic in `modules/NpuBench.psm1`.
- Python path: prompts loaded from `prompts.json`; evaluator at `tools/evaluate.py`; web UI under `tools/webui/`.

### Disclaimer
Token counts require backend-provided usage. If unavailable, counts can be computed with a local tokenizer (HF AutoTokenizer). Provide `-TokenizerModelId` or rely on auto-detection; for offline-only, use `-TokenizerLocalOnly` after caching the tokenizer locally.
