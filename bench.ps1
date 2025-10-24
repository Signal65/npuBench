param(
    [string]$ServerHost,
    [int]$Port,
    [string]$BaseUrl,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$BackendName,
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
    [switch]$DebugTokenizer
)

Import-Module -Force "$PSScriptRoot/modules/NpuBench.psm1"

function Guess-TokenizerModelId {
    param([string]$Model)
    $m = $Model.ToLowerInvariant()
    if ($m -match 'qwen') { return 'Qwen/Qwen2.5-7B-Instruct' }
    if ($m -match 'llama') { return 'meta-llama/Meta-Llama-3.1-8B-Instruct' }
    if ($m -match 'phi') { return 'microsoft/phi-3-mini-4k-instruct' }
    if ($m -match 'mistral') { return 'mistralai/Mistral-7B-Instruct-v0.2' }
    if ($m -match 'neox|gpt-j|gptj|gpt-neo') { return 'EleutherAI/gpt-neox-20b' }
    return $null
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
    if ($autoTok) {
        $TokenizerModelId = $autoTok
        Write-Host "Tokenizer fallback: using '$TokenizerModelId' (auto-detected from model name)"
    }
}
if ($forceTokenizer -and -not $TokenizerModelId) {
    Write-Host "Warning: backend '$BackendName' forces tokenizer fallback but no tokenizer model could be auto-detected. Consider passing -TokenizerModelId."
}

$sys = Get-SystemInfo

$BenchPrompts = @(
    [pscustomobject]@{ id='math_reasoning_1'; system='You are a helpful reasoning assistant.'; user='What is 73 * 49? Show concise reasoning, then the final answer.' }
)

$rows = New-Object System.Collections.Generic.List[object]
$totalRuns = $BenchPrompts.Count * $RunsPerPrompt
$runIndex = 0

foreach ($p in $BenchPrompts) {
    for ($i = 1; $i -le $RunsPerPrompt; $i++) {
        $runIndex++
        $messages = @()
        if ($p.system) { $messages += @{ role = 'system'; content = $p.system } }
        $messages += @{ role = 'user'; content = $p.user }

        $metrics = Invoke-OpenAIChatStream -BaseUrl $ResolvedBaseUrl -Model $Model -Messages $messages -MaxTokens $MaxOutputTokens -Temperature $Temperature -TopP $TopP -Seed $Seed -ApiKey $ApiKey -SkipCertificateCheck:$SkipCertificateCheck -TimeoutSec $TimeoutSec -AttemptUsageFallback -TokenizerModelId $TokenizerModelId -PythonPath $PythonPath -TokenizerLocalOnly:$TokenizerLocalOnly -ForceTokenizer:$forceTokenizer -DebugTokenizer:$DebugTokenizer

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
            prompt_id              = $p.id
            run_index              = $i
            is_warmup              = ($i -eq 1)
            token_source           = $tokenSource
            ttfb_ms                = $metrics.ttfb_ms
            prompt_tokens          = $metrics.prompt_tokens
            prompt_tokens_per_s    = $metrics.prompt_tokens_per_s
            completion_tokens      = $metrics.completion_tokens
            gen_tokens_per_s       = $metrics.gen_tokens_per_s
            total_time_ms          = $metrics.total_time_ms
            temperature            = $Temperature
            top_p                  = $TopP
            seed                   = $Seed
            runs_per_prompt        = $RunsPerPrompt
            max_output_tokens      = $MaxOutputTokens
        }
        $rows.Add($row) | Out-Null

        Write-Host ("[{0}/{1}] {2} (run {3}/{4}{5}) -> TTFB {6} ms, gen_tps {7}" -f $runIndex, $totalRuns, $p.id, $i, $RunsPerPrompt, ($(if($i -eq 1){' warmup'}else{''})), $metrics.ttfb_ms, $metrics.gen_tokens_per_s)
    }
}

if (-not (Test-Path -LiteralPath $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }
$timestamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$csvPath = Join-Path $OutputDir "bench_${timestamp}.csv"
$rows | Export-Csv -NoTypeInformation -Path $csvPath -Encoding utf8

Write-Host "Results saved to: $csvPath"
