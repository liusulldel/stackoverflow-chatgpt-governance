param(
    [ValidateSet("start", "status", "resume")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"

$baseDir = "D:\AI alignment\projects\stackoverflow_chatgpt_governance\raw\stackexchange_20251231"
$url = "https://archive.org/download/stackexchange_20251231/stackexchange_20251231/stackoverflow.com.7z"
$destination = Join-Path $baseDir "stackoverflow.com.7z"
$jobName = "stackoverflow_2025q4_dump"

New-Item -ItemType Directory -Force -Path $baseDir | Out-Null

function Show-JobStatus {
    $job = Get-BitsTransfer -Name $jobName -ErrorAction SilentlyContinue
    if (-not $job) {
        Write-Host "No BITS job named $jobName was found."
        return
    }

    $pct = $null
    if ($job.BytesTotal -gt 0 -and $job.BytesTotal -ne [UInt64]::MaxValue) {
        $pct = [math]::Round(100 * $job.BytesTransferred / $job.BytesTotal, 4)
    }

    [pscustomobject]@{
        DisplayName      = $job.DisplayName
        JobId            = $job.JobId
        JobState         = $job.JobState
        BytesTotal       = $job.BytesTotal
        BytesTransferred = $job.BytesTransferred
        Percent          = $pct
        Destination      = $destination
    } | Format-List
}

switch ($Action) {
    "start" {
        $existing = Get-BitsTransfer -Name $jobName -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Host "BITS job already exists:"
            Show-JobStatus
            break
        }

        $job = Start-BitsTransfer -Source $url -Destination $destination -Asynchronous -DisplayName $jobName
        Write-Host "Started BITS transfer."
        $job | Select-Object DisplayName, JobId, JobState, BytesTotal, BytesTransferred | Format-Table -AutoSize
    }
    "resume" {
        $job = Get-BitsTransfer -Name $jobName -ErrorAction SilentlyContinue
        if (-not $job) {
            Write-Host "No paused/suspended job found. Starting a new one."
            $job = Start-BitsTransfer -Source $url -Destination $destination -Asynchronous -DisplayName $jobName
        } else {
            Resume-BitsTransfer -BitsJob $job
        }
        Show-JobStatus
    }
    "status" {
        Show-JobStatus
    }
}
