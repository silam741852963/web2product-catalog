param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)

# Retry exit code
$retryExitCode = $env:RETRY_EXIT_CODE
if (-not $retryExitCode) { $retryExitCode = $env:DEEP_CRAWL_RETRY_EXIT_CODE }
if (-not $retryExitCode) { $retryExitCode = 17 }
$env:RETRY_EXIT_CODE = $retryExitCode
$env:DEEP_CRAWL_RETRY_EXIT_CODE = $retryExitCode

# Derive OUT_DIR from env or --out-dir arg
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

$retryFile = Join-Path $outDir "retry_companies.json"
$globalStateFile = Join-Path $outDir "crawl_global_state.json"

# Expected improvement + caps (new names with backward-compat)
$expectedMinImprovementRate = $env:EXPECTED_MIN_IMPROVEMENT_RATE
if (-not $expectedMinImprovementRate) { $expectedMinImprovementRate = $env:STRICT_MIN_RETRY_SUCCESS_RATE }
if (-not $expectedMinImprovementRate) { $expectedMinImprovementRate = 0.10 }

$maxRetryIter = $env:MAX_RETRY_ITER
if (-not $maxRetryIter) { $maxRetryIter = $env:STRICT_MAX_RETRY_ITER }
if (-not $maxRetryIter) { $maxRetryIter = 10 }

# OOM auto restart config
$oomRestartLimit = $env:OOM_RESTART_LIMIT
if (-not $oomRestartLimit) { $oomRestartLimit = 100 }
$oomRestartDelay = $env:OOM_RESTART_DELAY
if (-not $oomRestartDelay) { $oomRestartDelay = 10 }
$oomRestartCount = 0

# Persistent retry history file (JSONL)
$retryHistoryFile = if ($env:RETRY_HISTORY_FILE) { $env:RETRY_HISTORY_FILE } else { Join-Path $outDir "retry_history.jsonl" }
$historyDir = Split-Path $retryHistoryFile -Parent
if (-not (Test-Path $historyDir)) { New-Item -ItemType Directory -Force -Path $historyDir | Out-Null }

function Write-RetryHistory {
    param(
        [int]$Iteration,
        [int]$ExitCode,
        [string]$RetryMode,
        [string]$Phase,
        [int]$CurrentRetryCount,
        [int]$PrevRetryCount,
        [int]$TotalCompanies,
        [int]$AttemptedTotal,
        [string]$Reason
    )
    $obj = [pscustomobject]@{
        timestamp           = (Get-Date).ToString("o")
        iteration           = $Iteration
        exit_code           = $ExitCode
        retry_mode          = $RetryMode
        phase               = $Phase
        current_retry_count = $CurrentRetryCount
        prev_retry_count    = $PrevRetryCount
        total_companies     = $TotalCompanies
        attempted_total     = $AttemptedTotal
        reason              = $Reason
    }
    Add-Content -Path $retryHistoryFile -Value ($obj | ConvertTo-Json -Compress)
}

function Read-RetryInfo {
    # returns @{ retry_count=; total_companies=; attempted_total= }
    if (-not (Test-Path $retryFile)) {
        return @{ retry_count = 0; total_companies = 0; attempted_total = 0 }
    }
    try {
        $json = Get-Content $retryFile -Raw | ConvertFrom-Json
        $rc = 0
        if ($null -ne $json.retry_companies) { $rc = ($json.retry_companies | Measure-Object).Count }
        $total = 0
        if ($json.PSObject.Properties.Name -contains "total_companies") { $total = [int]$json.total_companies }
        $attempted = 0
        if ($json.PSObject.Properties.Name -contains "attempted_total") { $attempted = [int]$json.attempted_total }
        return @{ retry_count = $rc; total_companies = $total; attempted_total = $attempted }
    }
    catch {
        return @{ retry_count = 0; total_companies = 0; attempted_total = 0 }
    }
}

function Read-InProgressCount {
    if (-not (Test-Path $globalStateFile)) { return 0 }
    try {
        $json = Get-Content $globalStateFile -Raw | ConvertFrom-Json
        if ($null -eq $json.in_progress_companies) { return 0 }
        return ($json.in_progress_companies | Measure-Object).Count
    }
    catch { return 0 }
}

# Detect user-supplied --retry-mode once
$userRetryModeProvided = $false
$userRetryModeValue = $null
if ($RunArgs) {
    for ($i = 0; $i -lt $RunArgs.Length; $i++) {
        if ($RunArgs[$i] -eq "--retry-mode") {
            $userRetryModeProvided = $true
            if ($i + 1 -lt $RunArgs.Length) { $userRetryModeValue = $RunArgs[$i + 1] }
            break
        }
    }
}
if ($userRetryModeProvided) {
    Write-Host "[retry-wrapper] user-specified --retry-mode=$userRetryModeValue (wrapper will not override retry-mode)."
}

$iter = 1
$phase = "primary"  # primary -> retry
$retryIter = 0
$prevRetryCount = 0

while ($true) {
    $info = Read-RetryInfo
    $retryModeToUse = ""

    if ($userRetryModeProvided) {
        $retryModeToUse = "(user:$userRetryModeValue)"
    }
    else {
        if ($phase -eq "primary") {
            # first run uses "all" ONLY if retry list is empty or file missing
            if ([int]$info.retry_count -eq 0) { $retryModeToUse = "all" } else { $retryModeToUse = "skip-retry" }
        }
        else {
            $retryModeToUse = "only-retry"
        }
    }

    Write-Host "[retry-wrapper] iteration $iter phase=$phase retry_mode=$retryModeToUse"

    $cmdArgs = @()
    if (-not $userRetryModeProvided) { $cmdArgs += @("--retry-mode", $retryModeToUse) }
    if ($RunArgs) { $cmdArgs += $RunArgs }

    & python "scripts/run.py" @cmdArgs
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        $info = Read-RetryInfo
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -RetryMode $retryModeToUse -Phase $phase `
            -CurrentRetryCount ([int]$info.retry_count) -PrevRetryCount $prevRetryCount `
            -TotalCompanies ([int]$info.total_companies) -AttemptedTotal ([int]$info.attempted_total) -Reason "clean_exit"

        $inprog = Read-InProgressCount
        if ($inprog -gt 0) {
            Write-Host "[retry-wrapper] clean exit but crawl_global_state.json has in_progress_companies=$inprog"
            Write-Host "[retry-wrapper] running finalize pass (force-mark markdown completed for those companies; no LLM)."

            $finalArgs = @()
            if ($RunArgs) { $finalArgs += $RunArgs }
            $finalArgs += @("--finalize-in-progress-md", "--llm-mode", "none")

            & python "scripts/run.py" @finalArgs
            $finalExit = $LASTEXITCODE
            if ($finalExit -ne 0) {
                Write-Host "[retry-wrapper] finalize pass failed with exit_code=$finalExit; stopping."
                Write-RetryHistory -Iteration $iter -ExitCode $finalExit -RetryMode "finalize" -Phase "finalize" `
                    -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount `
                    -TotalCompanies ([int]$info.total_companies) -AttemptedTotal ([int]$info.attempted_total) -Reason "finalize_failed"
                exit $finalExit
            }
            Write-Host "[retry-wrapper] finalize pass completed."
        }

        Write-Host "[retry-wrapper] finished."
        exit 0
    }

    # Non-retry exit codes
    if ($exitCode -ne [int]$retryExitCode) {
        # OOM-ish: signal 9 => 137 on Linux; on Windows you won't see this normally.
        if ($exitCode -eq 137) {
            if ($oomRestartCount -lt [int]$oomRestartLimit) {
                $oomRestartCount += 1
                Write-Host "[retry-wrapper] exit 137 (likely OOM); restart $oomRestartCount/$oomRestartLimit after ${oomRestartDelay}s."
                $info = Read-RetryInfo
                Write-RetryHistory -Iteration $iter -ExitCode $exitCode -RetryMode $retryModeToUse -Phase $phase `
                    -CurrentRetryCount ([int]$info.retry_count) -PrevRetryCount $prevRetryCount `
                    -TotalCompanies ([int]$info.total_companies) -AttemptedTotal ([int]$info.attempted_total) -Reason "oom_restart"
                Start-Sleep -Seconds ([int]$oomRestartDelay)
                $iter += 1
                continue
            }
        }

        $info = Read-RetryInfo
        Write-Host "[retry-wrapper] run exited with non-retry code $exitCode; stopping."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -RetryMode $retryModeToUse -Phase $phase `
            -CurrentRetryCount ([int]$info.retry_count) -PrevRetryCount $prevRetryCount `
            -TotalCompanies ([int]$info.total_companies) -AttemptedTotal ([int]$info.attempted_total) -Reason "non_retry_exit"
        exit $exitCode
    }

    # RETRY_EXIT_CODE path
    if (-not (Test-Path $retryFile)) {
        Write-Host "[retry-wrapper] retry exit code but $retryFile not found; stopping."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -RetryMode $retryModeToUse -Phase $phase `
            -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -TotalCompanies 0 -AttemptedTotal 0 -Reason "retry_exit_missing_retry_file"
        exit 1
    }

    $info = Read-RetryInfo
    $currentRetryCount = [int]$info.retry_count
    $totalCompanies = [int]$info.total_companies
    $attemptedTotal = [int]$info.attempted_total

    Write-Host "[retry-wrapper] retry_companies=$currentRetryCount total_companies=$totalCompanies attempted_total=$attemptedTotal"

    $reason = "retry_exit_continue"

    if ($phase -eq "primary") {
        if ($totalCompanies -gt 0 -and $attemptedTotal -ge $totalCompanies -and $currentRetryCount -gt 0) {
            Write-Host "[retry-wrapper] attempted_total>=total_companies; switching to retry phase (only-retry)."
            $phase = "retry"
            $retryIter = 0
            $prevRetryCount = $currentRetryCount
        }
        else {
            Write-Host "[retry-wrapper] staying in primary phase."
        }

        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -RetryMode $retryModeToUse -Phase $phase `
            -CurrentRetryCount $currentRetryCount -PrevRetryCount $prevRetryCount `
            -TotalCompanies $totalCompanies -AttemptedTotal $attemptedTotal -Reason $reason

        $prevRetryCount = $currentRetryCount
        $iter += 1
        continue
    }

    # retry phase: expected improvement + max retry
    if ($prevRetryCount -eq 0) { $prevRetryCount = $currentRetryCount }

    if ($prevRetryCount -gt 0) {
        $succeeded = $prevRetryCount - $currentRetryCount
        $rate = 0.0
        if ($prevRetryCount -gt 0) { $rate = $succeeded / [double]$prevRetryCount }
        $percent = "{0:P1}" -f $rate
        Write-Host "[retry-wrapper] retry improvement: $succeeded/$prevRetryCount ($percent)"

        if ($currentRetryCount -eq 0) {
            $reason = "retry_exit_stop_empty"
        }
        elseif ($rate -lt [double]$expectedMinImprovementRate) {
            Write-Host "[retry-wrapper] progress below EXPECTED_MIN_IMPROVEMENT_RATE=$expectedMinImprovementRate; stopping."
            $reason = "retry_exit_stop_progress"
        }
    }

    if ($reason -eq "retry_exit_continue") {
        if ($retryIter -ge [int]$maxRetryIter) {
            Write-Host "[retry-wrapper] reached MAX_RETRY_ITER=$maxRetryIter; stopping."
            $reason = "retry_exit_stop_max_iter"
        }
    }

    Write-RetryHistory -Iteration $iter -ExitCode $exitCode -RetryMode $retryModeToUse -Phase $phase `
        -CurrentRetryCount $currentRetryCount -PrevRetryCount $prevRetryCount `
        -TotalCompanies $totalCompanies -AttemptedTotal $attemptedTotal -Reason $reason

    if ($reason -ne "retry_exit_continue") { exit 0 }

    $prevRetryCount = $currentRetryCount
    $retryIter += 1
    $iter += 1
}
