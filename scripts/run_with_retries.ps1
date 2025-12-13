param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)

# Retry exit code: prefer RETRY_EXIT_CODE, then DEEP_CRAWL_RETRY_EXIT_CODE, then default 17
$retryExitCode = $env:RETRY_EXIT_CODE
if (-not $retryExitCode) { $retryExitCode = $env:DEEP_CRAWL_RETRY_EXIT_CODE }
if (-not $retryExitCode) { $retryExitCode = 17 }

# Sync both env vars so run.py and wrapper agree
$env:RETRY_EXIT_CODE = $retryExitCode
$env:DEEP_CRAWL_RETRY_EXIT_CODE = $retryExitCode

# Derive OUT_DIR from env or --out-dir argument
$outDir = $env:OUT_DIR
if (-not $outDir) {
    $outDir = "outputs"
    if ($RunArgs) {
        for ($i = 0; $i -lt $RunArgs.Length; $i++) {
            if ($RunArgs[$i] -eq "--out-dir" -and ($i + 1 -lt $RunArgs.Length)) {
                $outDir = $RunArgs[$i + 1]
                break
            }
        }
    }
}
$env:OUT_DIR = $outDir

# Strict retry config
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

$retryFile = Join-Path $outDir "retry_companies.json"

$prevRetryCount = 0
$iter = 1

# Phase
# - primary  focus on non retry companies, skipping known retry ids
# - retry    focus on retry list only, with strict limits
$phase = "primary"
$strictRetryCount = 0

# Detect user supplied --retry-mode once
$userRetryModeProvided = $false
$userRetryModeValue = $null
if ($RunArgs) {
    for ($i = 0; $i -lt $RunArgs.Length; $i++) {
        if ($RunArgs[$i] -eq "--retry-mode") {
            $userRetryModeProvided = $true
            if ($i + 1 -lt $RunArgs.Length) {
                $userRetryModeValue = $RunArgs[$i + 1]
            }
            break
        }
    }
}
if ($userRetryModeProvided) {
    Write-Host "[retry-wrapper] user-specified --retry-mode=$userRetryModeValue (wrapper will not override retry-mode)."
}

while ($true) {
    Write-Host "[retry-wrapper] iteration $iter (phase=$phase)"

    # Decide retry mode for this iteration (only if user did not force one)
    $retryCompanyMode = "all"
    if (-not $userRetryModeProvided) {
        if ($phase -eq "primary") {
            if (Test-Path $retryFile) {
                # In primary phase, once we have a retry file, skip those IDs
                $retryCompanyMode = "skip-retry"
            }
        }
        else {
            # In retry phase we only work on the retry list
            $retryCompanyMode = "only-retry"
        }
        Write-Host "[retry-wrapper] RETRY_COMPANY_MODE=$retryCompanyMode"
    }
    else {
        Write-Host "[retry-wrapper] using user-specified --retry-mode=$userRetryModeValue"
    }

    # Build final argument list for run.py
    $cmdArgs = @()
    if (-not $userRetryModeProvided) {
        $cmdArgs += @("--retry-mode", $retryCompanyMode)
    }
    if ($RunArgs) {
        $cmdArgs += $RunArgs
    }

    # Run the crawler
    & python "scripts/run.py" @cmdArgs
    $exitCode = $LASTEXITCODE

    # ---------------- Clean exit path ----------------
    if ($exitCode -eq 0) {
        # No new retry ids were produced by this run.
        # Still check if older retry file has pending companies.
        $currentRetryCount = 0
        if (Test-Path $retryFile) {
            try {
                $json = Get-Content $retryFile -Raw | ConvertFrom-Json
                if ($null -ne $json.retry_companies) {
                    $currentRetryCount = ($json.retry_companies | Measure-Object).Count
                }
            }
            catch {
                $currentRetryCount = 0
            }
        }

        # current_retry_count field is 0 on clean exit
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount 0 -PrevRetryCount $prevRetryCount -Reason "clean_exit"

        if ($phase -eq "primary" -and $currentRetryCount -gt 0) {
            Write-Host "[retry-wrapper] primary phase finished (no more non retry companies to process in this file)"
            Write-Host "[retry-wrapper] switching to retry phase with $currentRetryCount stalled or timeout companies."
            $phase = "retry"
            $prevRetryCount = $currentRetryCount
            $iter += 1
            continue
        }

        if ($phase -eq "retry" -and $currentRetryCount -gt 0) {
            Write-Host "[retry-wrapper] run exited with 0 but retry_companies.json still lists $currentRetryCount companies  stopping anyway."
        }
        else {
            Write-Host "[retry-wrapper] run finished cleanly and no pending retry companies remain  stopping."
        }
        break
    }

    # ---------------- Non retry exit code path ----------------
    if ($exitCode -ne [int]$retryExitCode) {
        Write-Host "[retry-wrapper] run exited with non retry code $exitCode  stopping."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -Reason "non_retry_exit"
        exit $exitCode
    }

    # ---------------- RETRY_EXIT_CODE path ----------------
    if (-not (Test-Path $retryFile)) {
        Write-Host "[retry-wrapper] retry exit code but $retryFile not found  stopping."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -Reason "retry_exit_missing_retry_file"
        exit 1
    }

    # Read fields from retry_companies.json
    try {
        $json = Get-Content $retryFile -Raw | ConvertFrom-Json
    }
    catch {
        Write-Host "[retry-wrapper] failed to parse $retryFile  stopping."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -Reason "retry_exit_invalid_retry_file"
        exit 1
    }

    $currentRetryCount = 0
    if ($null -ne $json.retry_companies) {
        $currentRetryCount = ($json.retry_companies | Measure-Object).Count
    }

    $allAttemptedFlag = $false
    if ($json.PSObject.Properties.Name -contains "all_attempted") {
        if ($json.all_attempted) { $allAttemptedFlag = $true }
    }

    $totalCompanies = 0
    if ($json.PSObject.Properties.Name -contains "total_companies") {
        $totalCompanies = [int]$json.total_companies
    }

    $attemptedTotal = 0
    if ($json.PSObject.Properties.Name -contains "attempted_total") {
        $attemptedTotal = [int]$json.attempted_total
    }

    Write-Host "[retry-wrapper] $currentRetryCount companies need retry."
    Write-Host "[retry-wrapper] all_attempted=$allAttemptedFlag total_companies=$totalCompanies attempted_total=$attemptedTotal"

    $reason = "retry_exit_continue"

    # -------------- PHASE primary --------------
    if ($phase -eq "primary") {
        if ($totalCompanies -eq 0) {
            Write-Host "[retry-wrapper] primary phase run had no companies to process in this file  switching to retry phase."
            $phase = "retry"
        }
        else {
            if ($attemptedTotal -lt $totalCompanies) {
                Write-Host "[retry-wrapper] run did not attempt all primary companies ($attemptedTotal/$totalCompanies)  staying in primary phase."
            }
            else {
                Write-Host "[retry-wrapper] all primary companies for this run were attempted at least once  switching to retry phase."
                $phase = "retry"
            }
        }

        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount $currentRetryCount -PrevRetryCount $prevRetryCount -Reason $reason
        $prevRetryCount = $currentRetryCount
        $iter += 1
        continue
    }

    # -------------- PHASE retry (strict) --------------
    if ($prevRetryCount -eq 0) {
        $prevRetryCount = $currentRetryCount
    }

    if ($prevRetryCount -gt 0) {
        $succeeded = $prevRetryCount - $currentRetryCount
        $rate = 0.0
        if ($prevRetryCount -gt 0) {
            $rate = $succeeded / [double]$prevRetryCount
        }
        $percent = "{0:P1}" -f $rate
        Write-Host "[retry-wrapper] progress from last retry set  $succeeded/$prevRetryCount ($percent)"

        if ($currentRetryCount -eq 0) {
            Write-Host "[retry-wrapper] no companies left to retry  stopping."
            $reason = "retry_exit_stop_empty"
        }
        elseif ($rate -lt [double]$strictMinSuccessRate) {
            Write-Host "[retry-wrapper] progress below STRICT_MIN_RETRY_SUCCESS_RATE=$strictMinSuccessRate  stopping."
            $reason = "retry_exit_stop_progress"
        }
    }

    if ($reason -eq "retry_exit_continue") {
        if ($strictRetryCount -ge [int]$strictMaxRetryIter) {
            Write-Host "[retry-wrapper] reached STRICT_MAX_RETRY_ITER=$strictMaxRetryIter  stopping."
            $reason = "retry_exit_stop_max_iter"
        }
    }

    Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount $currentRetryCount -PrevRetryCount $prevRetryCount -Reason $reason

    if ($reason -ne "retry_exit_continue") {
        break
    }

    $prevRetryCount = $currentRetryCount
    $strictRetryCount += 1
    $iter += 1
}
