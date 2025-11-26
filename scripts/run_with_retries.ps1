param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)

# Basic config
$retryExitCode = $env:RETRY_EXIT_CODE
if (-not $retryExitCode) { $retryExitCode = 17 }

$outDir = $env:OUT_DIR
if (-not $outDir) { $outDir = "outputs" }

# Strict mode config: only applied once all companies have been attempted at least once
$strictMinSuccessRate = $env:STRICT_MIN_RETRY_SUCCESS_RATE
if (-not $strictMinSuccessRate) { $strictMinSuccessRate = 0.1 }

$strictMaxRetryIter = $env:STRICT_MAX_RETRY_ITER
if (-not $strictMaxRetryIter) { $strictMaxRetryIter = 10 }

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

# Strict mode state
$strictMode = $false
$strictRetryCount = 0

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

    $currentRetryCount = 0
    if ($json.retry_companies) {
        $currentRetryCount = ($json.retry_companies | Measure-Object).Count
    }

    $allAttemptedFlag = $false
    if ($null -ne $json.all_attempted -and $json.all_attempted) {
        $allAttemptedFlag = $true
    }

    Write-Host "[retry-wrapper] $currentRetryCount companies need retry."
    Write-Host "[retry-wrapper] all_attempted=$allAttemptedFlag"

    $reason = "retry_exit_continue"

    # Enable strict mode once all companies have been attempted at least once
    if (-not $strictMode -and $allAttemptedFlag) {
        $strictMode = $true
        $strictRetryCount = 0
        Write-Host "[retry-wrapper] all companies have been attempted at least once; enabling strict retry policy (min success rate=$strictMinSuccessRate, max strict retries=$strictMaxRetryIter)."
    }

    # In strict mode, once we have a previous retry count, require improvement
    if ($strictMode -and $prevRetryCount -gt 0) {
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
        elseif ($rate -lt [double]$strictMinSuccessRate) {
            Write-Host "[retry-wrapper] progress below STRICT_MIN_RETRY_SUCCESS_RATE=$strictMinSuccessRate; stopping."
            $reason = "retry_exit_stop_progress"
        }
    }

    # In strict mode, also enforce a hard cap on number of strict retries
    if ($strictMode -and $reason -eq "retry_exit_continue") {
        if ($strictRetryCount -ge [int]$strictMaxRetryIter) {
            Write-Host "[retry-wrapper] reached STRICT_MAX_RETRY_ITER=$strictMaxRetryIter; stopping."
            $reason = "retry_exit_stop_max_iter"
        }
    }

    # Default behavior when not in strict mode:
    # - No improvement required
    # - No retry limit
    # So we just loop while reason stays "retry_exit_continue".

    # Log this iteration's outcome
    Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount $currentRetryCount -PrevRetryCount $prevRetryCount -Reason $reason

    if ($reason -ne "retry_exit_continue") {
        break
    }

    $prevRetryCount = $currentRetryCount
    if ($strictMode) {
        $strictRetryCount += 1
    }
    $iter += 1
}