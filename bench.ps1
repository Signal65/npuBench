param(
    [string]$ServerHost,
    [int]$Port,
    [string]$BaseUrl,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$BackendName,
    [string]$PromptsFile = "prompts.json",
    [int]$RunsPerPrompt = 4,
    [Nullable[int]]$MaxOutputTokens = $null,
    [double]$Temperature = 0,
    [double]$TopP = 1,
    [int]$Seed = 0,
    [string]$ApiKey,
    [string]$OutputDir = "results",
    [switch]$SkipCertificateCheck,
    [int]$TimeoutSec = 600,
    [string]$TokenizerModelId,
    [string]$PythonPath = 'python',
    [switch]$TokenizerLocalOnly,
    [switch]$DebugTokenizer,
    [switch]$IncludeOptional
)

Import-Module -Force "$PSScriptRoot/modules/NpuBench.psm1"

function Guess-TokenizerModelId {
    param([string]$Model)
    # Prefer a direct HF repo id or local path if provided
    if ($Model -match '/') { return $Model }
    if (Test-Path -LiteralPath $Model) { return $Model }

    $m = $Model.ToLowerInvariant()
    if ($m -match 'qwen') { return 'Qwen/Qwen2.5-7B-Instruct' }
    if ($m -match 'llama') { return 'meta-llama/Meta-Llama-3.1-8B-Instruct' }
    if ($m -match 'phi') { return 'microsoft/phi-3-mini-4k-instruct' }
    if ($m -match 'mistral') { return 'mistralai/Mistral-7B-Instruct-v0.2' }
    if ($m -match 'glt-o') { return 'EleutherAI/gpt-neox-20b' }
    if ($m -match 'gemma') { return 'google/gemma-7b-it' }
    return $null
}

function Load-Prompts {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Prompts file not found: $Path" }
    $lines = Get-Content -LiteralPath $Path -Raw
    $items = @()
    # Try JSON first
    $parsed = $null
    try { $parsed = $lines | ConvertFrom-Json -ErrorAction Stop } catch { $parsed = $null }
    if ($parsed -ne $null) {
        # If root has a 'prompts' array, use it; else if it's already an array, use it; else treat as single prompt object
        if ($parsed.PSObject.Properties.Name -contains 'prompts' -and $parsed.prompts) {
            $items = @($parsed.prompts)
        } elseif ($parsed -is [System.Collections.IEnumerable] -and -not ($parsed -is [string])) {
            $items = @($parsed)
        } else {
            $items = @($parsed)
        }
    } else {
        # JSONL fallback
        $items = @()
        Get-Content -LiteralPath $Path | ForEach-Object {
            $obj = $null
            try { $obj = $_ | ConvertFrom-Json -ErrorAction Stop } catch { $obj = $null }
            if ($obj -ne $null) { $items += ,$obj }
        }
    }
    $out = New-Object System.Collections.Generic.List[object]
    $idx = 0
    foreach ($it in $items) {
        $idx++
        $id = $null
        if ($it -is [psobject]) { $id = $it.id }
        if (-not $id) { $id = "prompt_${idx}" }
        if ($it -is [psobject] -and $it.messages) {
            $out.Add([pscustomobject]@{ id=$id; category=$it.category; messages=$it.messages; max_output_tokens=$it.max_output_tokens }) | Out-Null
            continue
        }
        $system = $null
        if ($it -is [psobject] -and $it.system) { $system = [string]$it.system }
        $user = $null
        if ($it -is [psobject] -and $it.user) { $user = [string]$it.user }
        elseif ($it -is [psobject] -and $it.prompt) { $user = [string]$it.prompt }
        elseif ($it -is [psobject] -and $it.input) { $user = [string]$it.input }
        if (-not $user) { continue }
        $out.Add([pscustomobject]@{ id=$id; category=$it.category; system=$system; user=$user; max_output_tokens=$it.max_output_tokens }) | Out-Null
    }
    return ,$out
}

$ResolvedBaseUrl = New-OpenAIBaseUrl -BaseUrl $BaseUrl -ServerHost $ServerHost -Port $Port
if (-not (Test-OpenAIEndpoint -BaseUrl $ResolvedBaseUrl -ApiKey $ApiKey -SkipCertificateCheck:$SkipCertificateCheck)) {
    Write-Error "Endpoint check failed at $ResolvedBaseUrl. Ensure the server is reachable and OpenAI-compatible."
    exit 1
}

$forceTokenizer = $false
if ($BackendName -and $BackendName.ToLowerInvariant() -eq 'foundry') {
    $forceTokenizer = $true
}

if (-not $TokenizerModelId) {
    $autoTok = Guess-TokenizerModelId -Model $Model
    if (-not $autoTok) {
        $autoTok = Resolve-TokenizerModelId -BaseUrl $ResolvedBaseUrl -SelectedModelId $Model -ApiKey $ApiKey -SkipCertificateCheck:$SkipCertificateCheck -PythonPath $PythonPath -TokenizerLocalOnly:$TokenizerLocalOnly -DebugTokenizer:$DebugTokenizer
    }
    if ($autoTok) {
        $TokenizerModelId = $autoTok
        Write-Host "Tokenizer fallback: using '$TokenizerModelId' (auto-detected)"
    }
}
if ($forceTokenizer -and -not $TokenizerModelId) {
    Write-Host "Warning: backend '$BackendName' forces tokenizer fallback but no tokenizer model could be auto-detected. Consider passing -TokenizerModelId."
}

$sys = Get-SystemInfo

$BenchPrompts = @()
try {
    $BenchPrompts = Load-Prompts -Path $PromptsFile
    Write-Host ("Loaded {0} prompts from {1}" -f $BenchPrompts.Count, $PromptsFile)
} catch {
    Write-Error "Failed to load prompts: $($_.Exception.Message)"
    exit 1
}
# Make Intermediate/Substantial summarization optional unless explicitly included
if (-not $IncludeOptional) {
    $BenchPrompts = $BenchPrompts | Where-Object { $_.category -ne 'Summarization, Intermediate' -and $_.category -ne 'Summarization, Substantial' }
}
if (-not $BenchPrompts -or $BenchPrompts.Count -eq 0) {
    Write-Error "No prompts selected after filtering."
    exit 1
}

$rows = New-Object System.Collections.Generic.List[object]
$totalRuns = $BenchPrompts.Count * $RunsPerPrompt
$runIndex = 0

foreach ($p in $BenchPrompts) {
    for ($i = 1; $i -le $RunsPerPrompt; $i++) {
        $runIndex++
        $messages = @()
        if ($p.messages) {
            $messages = @()
            foreach ($m in $p.messages) { $messages += @{ role = [string]$m.role; content = [string]$m.content } }
        } else {
            if ($p.system) { $messages += @{ role = 'system'; content = $p.system } }
            $messages += @{ role = 'user'; content = $p.user }
        }

        $effMax = $MaxOutputTokens
        if ($p.max_output_tokens -ne $null) { $effMax = [int]$p.max_output_tokens }

        $metrics = Invoke-OpenAIChatStream -BaseUrl $ResolvedBaseUrl -Model $Model -Messages $messages -MaxTokens $effMax -Temperature $Temperature -TopP $TopP -Seed $Seed -ApiKey $ApiKey -SkipCertificateCheck:$SkipCertificateCheck -TimeoutSec $TimeoutSec -AttemptUsageFallback -TokenizerModelId $TokenizerModelId -PythonPath $PythonPath -TokenizerLocalOnly:$TokenizerLocalOnly -ForceTokenizer:$forceTokenizer -DebugTokenizer:$DebugTokenizer

        $tokenSource = if ($metrics.prompt_tokens -ne $null -and $metrics.completion_tokens -ne $null) { if ($TokenizerModelId) { 'usage_or_hf' } else { 'usage' } } else { if ($TokenizerModelId) { if ($forceTokenizer) { 'hf_failed_forced' } else { 'hf_failed' } } else { 'none' } }

        $row = [pscustomobject]@{
            timestamp              = (Get-Date).ToString('o')
            hostname               = $sys.hostname
            os_version             = $sys.os_version
            cpu_name               = $sys.cpu_name
            gpu_names              = $sys.gpu_names
            npu_names              = $sys.npu_names
            ram_gb                 = $sys.ram_gb
            backend_name           = $BackendName
            base_url               = $ResolvedBaseUrl
            model                  = $Model
            tokenizer_model_id     = $TokenizerModelId
            prompt_id              = $p.id
            run_index              = $i
            is_warmup              = ($i -eq 1)
            token_source           = $tokenSource
            ttft_ms                = $metrics.ttft_ms
            prompt_tokens          = $metrics.prompt_tokens
            prompt_tokens_per_s    = $metrics.prompt_tokens_per_s
            completion_tokens      = $metrics.completion_tokens
            gen_tokens_per_s       = $metrics.gen_tokens_per_s
            total_time_ms          = $metrics.total_time_ms
            temperature            = $Temperature
            top_p                  = $TopP
            seed                   = $Seed
            runs_per_prompt        = $RunsPerPrompt
            max_output_tokens      = $effMax
            completion_text        = $metrics.completion_text
        }
        $rows.Add($row) | Out-Null

        Write-Host ("[{0}/{1}] {2} (run {3}/{4}{5}) -> TTFT {6} ms, gen_tps {7}" -f $runIndex, $totalRuns, $p.id, $i, $RunsPerPrompt, ($(if($i -eq 1){' warmup'}else{''})), $metrics.ttft_ms, $metrics.gen_tokens_per_s)
    }
}

if (-not (Test-Path -LiteralPath $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }
$timestamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$csvPath = Join-Path $OutputDir "bench_${timestamp}.csv"
$rows | Export-Csv -NoTypeInformation -Path $csvPath -Encoding utf8

Write-Host "Results saved to: $csvPath"
