param(
    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Stop"

$baseDir = Split-Path -Parent $PSScriptRoot
$rawDir = Join-Path $baseDir "raw\stackexchange_20251231"
$processedDir = Join-Path $baseDir "processed"
$logPath = Join-Path $rawDir "panel_validation_followup.log"
$statePath = Join-Path $rawDir "panel_validation_followup_state.json"
$validatorScript = Join-Path $baseDir "scripts\validate_2025_dump_outputs.py"

$requiredPanelOutputs = @(
    (Join-Path $processedDir "stackexchange_20251231_question_level_enriched.parquet"),
    (Join-Path $processedDir "stackexchange_20251231_primary_panel.csv"),
    (Join-Path $processedDir "stackexchange_20251231_fractional_panel.csv"),
    (Join-Path $processedDir "stackexchange_20251231_panel_summary.json")
)

$requiredValidationOutputs = @(
    (Join-Path $processedDir "stackexchange_20251231_validation_report.json"),
    (Join-Path $processedDir "stackexchange_20251231_validation_report.md")
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
Write-Log "Panel validation follow-up watcher started."
Save-State -Status "waiting_for_panel_outputs" -Detail "Waiting for stackexchange_20251231 panel outputs."

while ($true) {
    $missingOutputs = @($requiredPanelOutputs | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missingOutputs.Count -eq 0) {
        break
    }
    $detail = "Panel outputs not ready yet; missing count=$($missingOutputs.Count)"
    Save-State -Status "waiting_for_panel_outputs" -Detail $detail
    Write-Log $detail
    Start-Sleep -Seconds $PollSeconds
}

Invoke-CheckedPython -StepStatus "validating_panel_outputs" -StepDetail "Running 2025 dump validation report builder." -ScriptPath $validatorScript -RequiredOutputs $requiredValidationOutputs

Save-State -Status "completed" -Detail "Panel validation outputs verified."
Write-Log "Panel validation follow-up watcher completed successfully."
