param(
    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Stop"

$jobName = "stackoverflow_2025q4_dump"
$baseDir = "D:\AI alignment\projects\stackoverflow_chatgpt_governance"
$rawDir = Join-Path $baseDir "raw\stackexchange_20251231"
$logPath = Join-Path $rawDir "watch_and_parse_2025_dump.log"
$statePath = Join-Path $rawDir "watch_and_parse_state.json"
$parseScript = Join-Path $baseDir "scripts\build_stackoverflow_2025_dump_extension.py"
$panelScript = Join-Path $baseDir "scripts\build_2025_dump_closure_panel.py"
$whoStillAnswersScript = Join-Path $baseDir "scripts\build_who_still_answers_analysis.py"
$processedDir = Join-Path $baseDir "processed"
$manifestPath = Join-Path $rawDir "stackexchange_20251231_manifest.json"
$whoStillAnswersMetadata = Join-Path $processedDir "who_still_answers_metadata.json"

$parseOutputs = @(
    $manifestPath,
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
Write-Log "Watcher started."
Save-State -Status "watching" -Detail "Waiting for BITS job to finish."

while ($true) {
    $job = Get-BitsTransfer -Name $jobName -ErrorAction SilentlyContinue
    if (-not $job) {
        Write-Log "BITS job not found. Sleeping."
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    if ($job.JobState -eq "Transferred") {
        Write-Log "BITS job reached Transferred. Completing transfer."
        Complete-BitsTransfer -BitsJob $job
        break
    }

    $pct = $null
    if ($job.BytesTotal -gt 0 -and $job.BytesTotal -ne [UInt64]::MaxValue) {
        $pct = [math]::Round(100 * $job.BytesTransferred / $job.BytesTotal, 4)
    }
    Save-State -Status "watching" -Detail "BITS state=$($job.JobState); percent=$pct"
    Write-Log "BITS state=$($job.JobState); percent=$pct"
    Start-Sleep -Seconds $PollSeconds
}

Invoke-CheckedPython -StepStatus "parsing" -StepDetail "Running dump parser." -ScriptPath $parseScript -RequiredOutputs $parseOutputs
Invoke-CheckedPython -StepStatus "building_panel" -StepDetail "Running closure panel builder." -ScriptPath $panelScript -RequiredOutputs $panelOutputs
Invoke-CheckedPython -StepStatus "building_contributor_paper" -StepDetail "Running contributor-reallocation analysis builder." -ScriptPath $whoStillAnswersScript -RequiredOutputs @($whoStillAnswersMetadata)

Save-State -Status "completed" -Detail "Dump parsed, closure panel built, and contributor-reallocation analysis refreshed."
Write-Log "Watcher completed successfully."
