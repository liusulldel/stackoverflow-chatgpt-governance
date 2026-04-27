param(
    [int]$ParserPid,
    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Stop"

$baseDir = Split-Path -Parent $PSScriptRoot
$rawDir = Join-Path $baseDir "raw\stackexchange_20251231"
$processedDir = Join-Path $baseDir "processed"
$logPath = Join-Path $rawDir "manual_parse_followup.log"
$statePath = Join-Path $rawDir "manual_parse_followup_state.json"
$panelScript = Join-Path $baseDir "scripts\build_2025_dump_closure_panel.py"

$parseOutputs = @(
    (Join-Path $processedDir "stackexchange_20251231_focal_questions.parquet"),
    (Join-Path $processedDir "stackexchange_20251231_focal_answers.parquet"),
    (Join-Path $processedDir "stackexchange_20251231_focal_accept_votes.parquet"),
    (Join-Path $processedDir "stackexchange_20251231_focal_summary.json")
)

$panelOutputs = @(
    (Join-Path $processedDir "stackexchange_20251231_question_level_enriched.parquet"),
    (Join-Path $processedDir "stackexchange_20251231_primary_panel.csv"),
    (Join-Path $processedDir "stackexchange_20251231_fractional_panel.csv"),
    (Join-Path $processedDir "stackexchange_20251231_panel_summary.json")
)

function Write-Log {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("o")
    Add-Content -Path $logPath -Value "[$timestamp] $Message"
}

function Save-State {
    param(
        [string]$Status,
        [string]$Detail
    )
    [pscustomobject]@{
        status = $Status
        detail = $Detail
        parser_pid = $ParserPid
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Path $statePath -Encoding UTF8
}

function Invoke-CheckedPython {
    param(
        [string]$StepStatus,
        [string]$StepDetail,
        [string]$ScriptPath,
        [string[]]$RequiredOutputs = @()
    )

    Save-State -Status $StepStatus -Detail $StepDetail
    Write-Log $StepDetail

    & python $ScriptPath 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $message = "$StepDetail failed with exit code $exitCode."
        Save-State -Status "failed" -Detail $message
        Write-Log $message
        throw $message
    }

    foreach ($requiredOutput in $RequiredOutputs) {
        if (-not (Test-Path -LiteralPath $requiredOutput)) {
            $message = "$StepDetail finished with exit code 0 but missing expected output: $requiredOutput"
            Save-State -Status "failed" -Detail $message
            Write-Log $message
            throw $message
        }
    }
}

New-Item -ItemType Directory -Force -Path $rawDir | Out-Null
Write-Log "Manual parse follow-up watcher started."
Save-State -Status "watching_parser" -Detail "Waiting for parser PID $ParserPid to finish."

while ($true) {
    $process = Get-Process -Id $ParserPid -ErrorAction SilentlyContinue
    if (-not $process) {
        break
    }

    $detail = "Parser still running; CPU=$([math]::Round($process.CPU, 2)); WS=$([math]::Round($process.WorkingSet64 / 1GB, 2)) GB"
    Save-State -Status "watching_parser" -Detail $detail
    Write-Log $detail
    Start-Sleep -Seconds $PollSeconds
}

$missingParseOutputs = @($parseOutputs | Where-Object { -not (Test-Path -LiteralPath $_) })
if ($missingParseOutputs.Count -gt 0) {
    $message = "Parser process ended but parse outputs are missing: $($missingParseOutputs -join '; ')"
    Save-State -Status "failed" -Detail $message
    Write-Log $message
    throw $message
}

Invoke-CheckedPython -StepStatus "building_panel" -StepDetail "Running closure panel builder after manual parser completion." -ScriptPath $panelScript -RequiredOutputs $panelOutputs

Save-State -Status "completed" -Detail "Manual parser follow-up completed and closure panel outputs verified."
Write-Log "Manual parse follow-up watcher completed successfully."
