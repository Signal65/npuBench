function Ensure-NpuBenchNetSupport {
    try { $null = [System.Net.Http.HttpClient] } catch {
        try { Add-Type -AssemblyName 'System.Net.Http' -ErrorAction Stop } catch {
            try { [System.Reflection.Assembly]::Load('System.Net.Http') | Out-Null } catch { }
        }
    }
    try {
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor [System.Net.SecurityProtocolType]::Tls12
    } catch { }
    try {
        $tls13 = [enum]::Parse([System.Net.SecurityProtocolType], 'Tls13')
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor $tls13
    } catch { }
}

function Invoke-HFTokenCount {
    param(
        [Parameter(Mandatory=$true)][string]$ModelId,
        [Parameter(Mandatory=$true)][object[]]$Messages,
        [string]$CompletionText,
        [string]$PythonPath = 'python',
        [switch]$LocalFilesOnly,
        [switch]$DebugTokenizer
    )
    $scriptPath = Join-Path $PSScriptRoot '../tools/token_count.py'
    $messagesJson = ($Messages | ConvertTo-Json -Depth 8)

    $tmpDir = [System.IO.Path]::GetTempPath()
    $msgPath = Join-Path $tmpDir ("npuBench_messages_" + [Guid]::NewGuid().ToString() + ".json")
    [System.IO.File]::WriteAllText($msgPath, $messagesJson, [System.Text.Encoding]::UTF8)

    $compPath = $null
    if ($CompletionText) {
        $compPath = Join-Path $tmpDir ("npuBench_completion_" + [Guid]::NewGuid().ToString() + ".txt")
        [System.IO.File]::WriteAllText($compPath, $CompletionText, [System.Text.Encoding]::UTF8)
    }

    $msgPathTrim = $msgPath.Trim()
    $compPathTrim = if ($compPath) { $compPath.Trim() } else { $null }

    # Build arguments without wrapping paths in quotes (paths have no spaces in Temp)
    $argParts = @($scriptPath, '--model-id', $ModelId, '--messages-path', $msgPathTrim)
    if ($compPathTrim) { $argParts += @('--completion-path', $compPathTrim) }
    if ($LocalFilesOnly) { $argParts += '--local-files-only' }
    $argString = ($argParts -join ' ')

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonPath
    $psi.Arguments = $argString
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    if ($DebugTokenizer) {
        Write-Host ("[tokenizer] argv: {0}" -f ($argParts -join ' | '))
        Write-Host ("[tokenizer] cmd: {0} {1}" -f $PythonPath, $psi.Arguments)
        Write-Host ("[tokenizer] model: {0}, local_only: {1}" -f $ModelId, [bool]$LocalFilesOnly)
        $compDisplay = if ($compPathTrim) { $compPathTrim } else { '' }
        Write-Host ("[tokenizer] messages_file: {0}, completion_file: {1}" -f $msgPathTrim, $compDisplay)
    }

    $stdout = ''
    $stderr = ''
    $exitCode = -1
    try {
        $p = New-Object System.Diagnostics.Process
        $p.StartInfo = $psi
        [void]$p.Start()
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()
        $exitCode = $p.ExitCode
    } catch {
        $stderr = "process start failed: $($_.Exception.Message)"
    } finally {
        try { if (Test-Path -LiteralPath $msgPath) { Remove-Item -LiteralPath $msgPath -Force } } catch { }
        try { if ($compPath -and (Test-Path -LiteralPath $compPath)) { Remove-Item -LiteralPath $compPath -Force } } catch { }
    }

    if ($DebugTokenizer) {
        Write-Host ("[tokenizer] exit: {0}" -f $exitCode)
        if ($stdout) { Write-Host ("[tokenizer] stdout: {0}" -f ($stdout.Substring(0, [Math]::Min($stdout.Length, 1000)))) }
        if ($stderr) { Write-Host ("[tokenizer] stderr: {0}" -f ($stderr.Substring(0, [Math]::Min($stderr.Length, 1000)))) }
    }

    if ($exitCode -ne 0) { return $null }
    try { return $stdout | ConvertFrom-Json -ErrorAction Stop } catch { return $null }
}

function Get-SystemInfo {
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $gpus = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name -ErrorAction SilentlyContinue
    $gpusStr = if ($gpus) { ($gpus -join '; ') } else { '' }

    $npuMatches = Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match 'NPU|Neural|VPU|TPU|NPX|Neural Processing'
    } | Select-Object -ExpandProperty Name -ErrorAction SilentlyContinue
    $npusStr = if ($npuMatches) { ($npuMatches -join '; ') } else { '' }

    $ramBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    $ramGb = [Math]::Round($ramBytes / 1GB, 1)

    [pscustomobject]@{
        hostname   = $env:COMPUTERNAME
        os_version = "Windows $($os.Version) ($($os.Caption))"
        cpu_name   = $cpu.Name
        gpu_names  = $gpusStr
        npu_names  = $npusStr
        ram_gb     = $ramGb
    }
}

function New-OpenAIBaseUrl {
    param(
        [string]$BaseUrl,
        [string]$ServerHost,
        [int]$Port
    )
    if (-not [string]::IsNullOrWhiteSpace($BaseUrl)) {
        if ($BaseUrl.TrimEnd('/') -match '/v1$') { return $BaseUrl.TrimEnd('/') }
        return ($BaseUrl.TrimEnd('/') + '/v1')
    }
    if (-not $ServerHost -or -not $Port) { throw 'Either BaseUrl or ServerHost and Port must be provided.' }
    "http://$ServerHost`:$Port/v1"
}

function New-HttpClient {
    param(
        [switch]$SkipCertificateCheck,
        [string]$ApiKey,
        [int]$TimeoutSec = 600
    )
    Ensure-NpuBenchNetSupport
    $handler = New-Object System.Net.Http.HttpClientHandler
    if ($SkipCertificateCheck) {
        try {
            $handler.ServerCertificateCustomValidationCallback = { param($sender,$cert,$chain,$errors) return $true }
        } catch {
            try { [System.Net.ServicePointManager]::ServerCertificateValidationCallback = { param($sender,$cert,$chain,$errors) return $true } } catch { }
        }
    }
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
    $client.DefaultRequestHeaders.Accept.Clear() | Out-Null
    $mt = New-Object System.Net.Http.Headers.MediaTypeWithQualityHeaderValue('text/event-stream')
    $client.DefaultRequestHeaders.Accept.Add($mt)
    if ($ApiKey) {
        $auth = New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $ApiKey)
        $client.DefaultRequestHeaders.Authorization = $auth
    }
    $client
}

function Test-OpenAIEndpoint {
    param(
        [string]$BaseUrl,
        [string]$ApiKey,
        [switch]$SkipCertificateCheck,
        [int]$TimeoutSec = 60
    )
    $client = New-HttpClient -SkipCertificateCheck:$SkipCertificateCheck -ApiKey $ApiKey -TimeoutSec $TimeoutSec
    try {
        $uri = "$BaseUrl/models"
        $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Get, $uri)
        $req.Headers.Accept.Clear() | Out-Null
        $mt = New-Object System.Net.Http.Headers.MediaTypeWithQualityHeaderValue('application/json')
        $req.Headers.Accept.Add($mt)
        $resp = $client.SendAsync($req).GetAwaiter().GetResult()
        return $resp.IsSuccessStatusCode
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Get-OpenAICompletionData {
    param(
        [string]$BaseUrl,
        [string]$Model,
        [object[]]$Messages,
        [Nullable[int]]$MaxTokens,
        [double]$Temperature = 0,
        [double]$TopP = 1,
        [int]$Seed,
        [string]$ApiKey,
        [switch]$SkipCertificateCheck,
        [int]$TimeoutSec = 600
    )
    Ensure-NpuBenchNetSupport
    $handler = New-Object System.Net.Http.HttpClientHandler
    if ($SkipCertificateCheck) {
        try { $handler.ServerCertificateCustomValidationCallback = { param($sender,$cert,$chain,$errors) return $true } } catch { }
    }
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
    if ($ApiKey) { $client.DefaultRequestHeaders.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $ApiKey) }

    try {
        $uri = "$BaseUrl/chat/completions"
        $body = [ordered]@{
            model       = $Model
            messages    = $Messages
            stream      = $false
            temperature = $Temperature
            top_p       = $TopP
        }
        if ($MaxTokens -ne $null) { $body.max_tokens = $MaxTokens }
        if ($Seed -ne $null) { $body.seed = $Seed }
        $json = ($body | ConvertTo-Json -Depth 8)
        $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, $uri)
        $req.Content = New-Object System.Net.Http.StringContent($json, [System.Text.Encoding]::UTF8, 'application/json')
        $resp = $client.SendAsync($req).GetAwaiter().GetResult()
        $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if (-not $resp.IsSuccessStatusCode) { return $null }
        $obj = $text | ConvertFrom-Json -ErrorAction SilentlyContinue
        if (-not $obj) { return $null }
        $choice = $null
        if ($obj.choices) { $choice = $obj.choices | Select-Object -First 1 }
        $content = $null
        if ($choice) {
            if ($choice.message -and $choice.message.content) { $content = [string]$choice.message.content }
            elseif ($choice.text) { $content = [string]$choice.text }
        }
        [pscustomobject]@{ text = $content; usage = $obj.usage }
    } catch {
        return $null
    } finally {
        $client.Dispose()
    }
}

function Get-OpenAIUsage {
    param(
        [string]$BaseUrl,
        [string]$Model,
        [object[]]$Messages,
        [Nullable[int]]$MaxTokens,
        [double]$Temperature = 0,
        [double]$TopP = 1,
        [int]$Seed,
        [string]$ApiKey,
        [switch]$SkipCertificateCheck,
        [int]$TimeoutSec = 600,
        [string]$TokenizerModelId,
        [string]$PythonPath = 'python',
        [switch]$TokenizerLocalOnly,
        [switch]$DebugTokenizer
    )
    Ensure-NpuBenchNetSupport
    $handler = New-Object System.Net.Http.HttpClientHandler
    if ($SkipCertificateCheck) {
        try {
            $handler.ServerCertificateCustomValidationCallback = { param($sender,$cert,$chain,$errors) return $true }
        } catch {
            try { [System.Net.ServicePointManager]::ServerCertificateValidationCallback = { param($sender,$cert,$chain,$errors) return $true } } catch { }
        }
    }
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
    if ($ApiKey) {
        $auth = New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $ApiKey)
        $client.DefaultRequestHeaders.Authorization = $auth
    }

    try {
        $uri = "$BaseUrl/chat/completions"
        $body = [ordered]@{
            model       = $Model
            messages    = $Messages
            stream      = $false
            temperature = $Temperature
            top_p       = $TopP
        }
        if ($MaxTokens -ne $null) { $body.max_tokens = $MaxTokens }
        if ($Seed -ne $null) { $body.seed = $Seed }
        $json = ($body | ConvertTo-Json -Depth 8)
        $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, $uri)
        $mt = New-Object System.Net.Http.Headers.MediaTypeWithQualityHeaderValue('application/json')
        $req.Headers.Accept.Add($mt)
        $content = New-Object System.Net.Http.StringContent($json, [System.Text.Encoding]::UTF8, 'application/json')
        $req.Content = $content
        $resp = $client.SendAsync($req).GetAwaiter().GetResult()
        $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if (-not $resp.IsSuccessStatusCode) { throw 'non-success' }
        $obj = $text | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($obj -and $obj.usage) { return $obj.usage }
        throw 'no-usage'
    } catch {
        if (-not [string]::IsNullOrWhiteSpace($TokenizerModelId)) {
            $fallback = Invoke-HFTokenCount -ModelId $TokenizerModelId -Messages $Messages -PythonPath $PythonPath -LocalFilesOnly:$TokenizerLocalOnly -DebugTokenizer:$DebugTokenizer
            if ($fallback) { return $fallback }
        }
        return $null
    } finally {
        $client.Dispose()
    }
}

function Invoke-OpenAIChatStream {
    param(
        [string]$BaseUrl,
        [string]$Model,
        [object[]]$Messages,
        [Nullable[int]]$MaxTokens,
        [double]$Temperature = 0,
        [double]$TopP = 1,
        [int]$Seed,
        [string]$ApiKey,
        [switch]$SkipCertificateCheck,
        [int]$TimeoutSec = 600,
        [switch]$AttemptUsageFallback,
        [string]$TokenizerModelId,
        [string]$PythonPath = 'python',
        [switch]$TokenizerLocalOnly,
        [switch]$ForceTokenizer,
        [switch]$DebugTokenizer
    )

    $client = New-HttpClient -SkipCertificateCheck:$SkipCertificateCheck -ApiKey $ApiKey -TimeoutSec $TimeoutSec

    try {
        $uri = "$BaseUrl/chat/completions"
        $body = [ordered]@{
            model           = $Model
            messages        = $Messages
            stream          = $true
            temperature     = $Temperature
            top_p           = $TopP
            stream_options  = @{ include_usage = $true }
        }
        if ($MaxTokens -ne $null) { $body.max_tokens = $MaxTokens }
        if ($Seed -ne $null) { $body.seed = $Seed }
        $json = ($body | ConvertTo-Json -Depth 8)

        $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, $uri)
        $content = New-Object System.Net.Http.StringContent($json, [System.Text.Encoding]::UTF8, 'application/json')
        $req.Content = $content
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        $stream = $resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $reader = New-Object System.IO.StreamReader($stream)

        $firstTokenMs = $null
        $completionText = New-Object System.Text.StringBuilder
        $promptTokens = $null
        $completionTokens = $null

        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if (-not $line) { continue }
            if (-not $line.StartsWith('data:')) { continue }
            $data = $line.Substring(5).Trim()
            if ($data -eq '[DONE]') { break }
            $obj = $null
            try { $obj = $data | ConvertFrom-Json -ErrorAction Stop } catch { continue }

            if ($null -ne $obj.usage) {
                if ($null -ne $obj.usage.prompt_tokens) { $promptTokens = [int]$obj.usage.prompt_tokens }
                if ($null -ne $obj.usage.completion_tokens) { $completionTokens = [int]$obj.usage.completion_tokens }
                if ($null -eq $completionTokens -and $null -ne $obj.usage.output_tokens) { $completionTokens = [int]$obj.usage.output_tokens }
                if ($null -eq $promptTokens -and $null -ne $obj.usage.input_tokens) { $promptTokens = [int]$obj.usage.input_tokens }
            } elseif ($null -ne $obj.event -and ($obj.event -like '*usage*') -and $null -ne $obj.data) {
                if ($null -eq $promptTokens -and $null -ne $obj.data.input_tokens) { $promptTokens = [int]$obj.data.input_tokens }
                if ($null -eq $completionTokens -and $null -ne $obj.data.output_tokens) { $completionTokens = [int]$obj.data.output_tokens }
            }

            if ($obj.choices) {
                foreach ($ch in $obj.choices) {
                    $delta = $ch.delta
                    $added = $null
                    if ($delta -and $delta.content) { $added = [string]$delta.content }
                    elseif ($ch.text) { $added = [string]$ch.text }
                    elseif ($ch.message -and $ch.message.content) { $added = [string]$ch.message.content }
                    elseif ($obj.delta -and $obj.delta.content) { $added = [string]$obj.delta.content }
                    if ($added) {
                        if ($null -eq $firstTokenMs) { $firstTokenMs = [int][Math]::Round($sw.Elapsed.TotalMilliseconds) }
                        [void]$completionText.Append($added)
                    }
                }
            }
        }
        $totalMs = [int][Math]::Round($sw.Elapsed.TotalMilliseconds)
        $reader.Dispose(); $stream.Dispose(); $resp.Dispose();

        if ($ForceTokenizer) {
            if (($completionText.Length -eq 0) -and $TokenizerModelId) {
                $comp = Get-OpenAICompletionData -BaseUrl $BaseUrl -Model $Model -Messages $Messages -MaxTokens $MaxTokens -Temperature $Temperature -TopP $TopP -Seed $Seed -ApiKey $ApiKey -SkipCertificateCheck:$SkipCertificateCheck -TimeoutSec $TimeoutSec
                if ($comp -and $comp.text) { [void]$completionText.Append($comp.text) }
            }
            if ($TokenizerModelId) {
                $fallback = Invoke-HFTokenCount -ModelId $TokenizerModelId -Messages $Messages -CompletionText $completionText.ToString() -PythonPath $PythonPath -LocalFilesOnly:$TokenizerLocalOnly -DebugTokenizer:$DebugTokenizer
                if ($fallback) {
                    $promptTokens = [int]$fallback.prompt_tokens
                    $completionTokens = [int]$fallback.completion_tokens
                    if ($DebugTokenizer) { Write-Host ("[tokenizer] counts: prompt={0}, completion={1}" -f $promptTokens, $completionTokens) }
                }
            }
        } elseif ($AttemptUsageFallback) {
            if ($null -eq $promptTokens -or $null -eq $completionTokens) {
                $usage = Get-OpenAIUsage -BaseUrl $BaseUrl -Model $Model -Messages $Messages -MaxTokens $MaxTokens -Temperature $Temperature -TopP $TopP -Seed $Seed -ApiKey $ApiKey -SkipCertificateCheck:$SkipCertificateCheck -TimeoutSec $TimeoutSec -TokenizerModelId $TokenizerModelId -PythonPath $PythonPath -TokenizerLocalOnly:$TokenizerLocalOnly -DebugTokenizer:$DebugTokenizer
                if ($usage) {
                    if ($null -eq $promptTokens -and $null -ne $usage.prompt_tokens) { $promptTokens = [int]$usage.prompt_tokens }
                    if ($null -eq $completionTokens -and $null -ne $usage.completion_tokens) { $completionTokens = [int]$usage.completion_tokens }
                }
            }
            if (($completionText.Length -eq 0) -and $TokenizerModelId) {
                $comp = Get-OpenAICompletionData -BaseUrl $BaseUrl -Model $Model -Messages $Messages -MaxTokens $MaxTokens -Temperature $Temperature -TopP $TopP -Seed $Seed -ApiKey $ApiKey -SkipCertificateCheck:$SkipCertificateCheck -TimeoutSec $TimeoutSec
                if ($comp -and $comp.text) { [void]$completionText.Append($comp.text) }
            }
            if (($null -eq $promptTokens -or $null -eq $completionTokens) -and $TokenizerModelId) {
                $fallback = Invoke-HFTokenCount -ModelId $TokenizerModelId -Messages $Messages -CompletionText $completionText.ToString() -PythonPath $PythonPath -LocalFilesOnly:$TokenizerLocalOnly -DebugTokenizer:$DebugTokenizer
                if ($fallback) {
                    if ($null -eq $promptTokens -and $null -ne $fallback.prompt_tokens) { $promptTokens = [int]$fallback.prompt_tokens }
                    if ($null -eq $completionTokens -and $null -ne $fallback.completion_tokens) { $completionTokens = [int]$fallback.completion_tokens }
                    if ($DebugTokenizer) { Write-Host ("[tokenizer] counts: prompt={0}, completion={1}" -f $promptTokens, $completionTokens) }
                }
            }
        }

        $ttfbMs = if ($firstTokenMs -ne $null) { [int]$firstTokenMs } else { $totalMs }
        $prefillSec = [Math]::Max($ttfbMs / 1000.0, 0.0001)
        $genSec = [Math]::Max(($totalMs - $ttfbMs) / 1000.0, 0.0001)

        $promptTps = $null
        $genTps = $null
        if ($promptTokens -ne $null) { $promptTps = [Math]::Round($promptTokens / $prefillSec, 3) }
        if ($completionTokens -ne $null) { $genTps = $null; if ($genSec -gt 0) { $genTps = [Math]::Round($completionTokens / $genSec, 3) } }

        [pscustomobject]@{
            ttft_ms                 = $ttfbMs
            total_time_ms           = $totalMs
            prompt_tokens           = $promptTokens
            completion_tokens       = $completionTokens
            prompt_tokens_per_s     = $promptTps
            gen_tokens_per_s        = $genTps
            completion_text_preview = $completionText.ToString().Substring(0, [Math]::Min($completionText.Length, 200))
        }
    } finally {
        $client.Dispose()
    }
}

Export-ModuleMember -Function Get-SystemInfo, New-OpenAIBaseUrl, Test-OpenAIEndpoint, Invoke-OpenAIChatStream, Get-OpenAIUsage, Get-OpenAICompletionData
