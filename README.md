## npuBench - Windows NPU LLM Benchmark (PowerShell)

A lightweight PowerShell framework to benchmark OpenAI-compatible LLM backends (Nexa, FastFlowLM, etc.) running on NPU-enabled Windows machines.

- Backend-agnostic: targets any OpenAI-compatible endpoint
- Measures: time to first token (TTFT), prompt tokens/sec, generation tokens/sec
- Captures device info: OS, CPU, GPU(s), NPU(s), RAM
- Outputs a single CSV per run with metrics and context

### Requirements
- Windows PowerShell 5.1 or PowerShell 7+
- Network access to an OpenAI-compatible server

### Installation
Clone or copy this folder. No external dependencies required.

Optional (for local tokenizer fallback when the server does not return usage):
- Python 3.8+
- Install packages:
```
python -m pip install transformers tokenizers jinja2
```
Notes:
- PyTorch/TF/Flax are NOT required for token counting; tokenizers work without them.
- If running offline, cache tokenizers first or use `-TokenizerLocalOnly` after caching.

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
- Add or modify baked prompts in `bench.ps1` under `$BenchPrompts`.
- The core HTTP and metrics logic is in `modules/NpuBench.psm1`.

### Disclaimer
Token counts require backend-provided usage. If unavailable, counts can be computed with a local tokenizer (HF AutoTokenizer). Provide `-TokenizerModelId` or rely on auto-detection; for offline-only, use `-TokenizerLocalOnly` after caching the tokenizer locally.
