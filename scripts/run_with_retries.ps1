param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)

$retryExitCode = $env:RETRY_EXIT_CODE
if (-not $retryExitCode) { $retryExitCode = 17 }

$maxRetryIter = $env:MAX_RETRY_ITER
if (-not $maxRetryIter) { $maxRetryIter = 10 }

$minSuccessRate = $env:MIN_RETRY_SUCCESS_RATE
if (-not $minSuccessRate) { $minSuccessRate = 0.3 }

$outDir = $env:OUT_DIR
if (-not $outDir) { $outDir = "outputs" }

# Persistent retry history file (JSONL)
$retryHistoryFile = if ($env:RETRY_HISTORY_FILE) {
    $env:RETRY_HISTORY_FILE
}
else {
    Join-Path $outDir "retry_history.jsonl"
}
$historyDir = Split-Path $retryHistoryFile -Parent
if (-not (Test-Path $historyDir)) {
    New-Item -ItemType Directory -Force -Path $historyDir | Out-Null
}

function Write-RetryHistory {
    param(
        [int]$Iteration,
        [int]$ExitCode,
        [int]$CurrentRetryCount,
        [int]$PrevRetryCount,
        [string]$Reason
    )

    $ts = (Get-Date).ToString("o")
    $obj = [pscustomobject]@{
        timestamp           = $ts
        iteration           = $Iteration
        exit_code           = $ExitCode
        current_retry_count = $CurrentRetryCount
        prev_retry_count    = $PrevRetryCount
        reason              = $Reason
    }
    $jsonLine = $obj | ConvertTo-Json -Compress
    Add-Content -Path $retryHistoryFile -Value $jsonLine
}

$prevRetryCount = 0
$iter = 1

while ($true) {
    Write-Host "[retry-wrapper] iteration ${iter}: running crawler..."
    & python scripts/run.py @RunArgs
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host "[retry-wrapper] run finished cleanly (exit 0); stopping."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount 0 -PrevRetryCount $prevRetryCount -Reason "clean_exit"
        break
    }

    if ($exitCode -ne [int]$retryExitCode) {
        Write-Host "[retry-wrapper] run exited with non-retry code $exitCode; stopping."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -Reason "non_retry_exit"
        exit $exitCode
    }

    $retryFile = Join-Path $outDir "retry_companies.json"
    if (-not (Test-Path $retryFile)) {
        Write-Host "[retry-wrapper] retry exit code but $retryFile not found; stopping."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -Reason "retry_exit_missing_retry_file"
        exit 1
    }

    $json = Get-Content $retryFile -Raw | ConvertFrom-Json
    $currentRetryCount = ($json.retry_companies | Measure-Object).Count
    Write-Host "[retry-wrapper] $currentRetryCount companies need retry."

    $reason = "retry_exit_continue"

    if ($prevRetryCount -gt 0) {
        $succeeded = $prevRetryCount - $currentRetryCount
        $rate = 0.0
        if ($prevRetryCount -gt 0) {
            $rate = $succeeded / [double]$prevRetryCount
        }
        $percent = "{0:P1}" -f $rate
        Write-Host "[retry-wrapper] progress from last retry set: $succeeded / $prevRetryCount ($percent)"

        if ($currentRetryCount -eq 0) {
            Write-Host "[retry-wrapper] no companies left to retry; stopping."
            $reason = "retry_exit_stop_empty"
        }
        elseif ($rate -lt [double]$minSuccessRate) {
            Write-Host "[retry-wrapper] progress below MIN_RETRY_SUCCESS_RATE=$minSuccessRate; stop auto retries."
            $reason = "retry_exit_stop_progress"
        }
    }

    if ($reason -eq "retry_exit_continue" -and $iter -ge [int]$maxRetryIter) {
        Write-Host "[retry-wrapper] reached MAX_RETRY_ITER=$maxRetryIter; stopping."
        $reason = "retry_exit_stop_max_iter"
    }

    # Log this iteration's outcome
    Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount $currentRetryCount -PrevRetryCount $prevRetryCount -Reason $reason

    if ($reason -ne "retry_exit_continue") {
        break
    }

    $prevRetryCount = $currentRetryCount
    $iter += 1
}