param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$RunArgs
)

# Retry exit code: prefer RETRY_EXIT_CODE, then DEEP_CRAWL_RETRY_EXIT_CODE, else 17
$retryExitCode = $env:RETRY_EXIT_CODE
if (-not $retryExitCode) { $retryExitCode = $env:DEEP_CRAWL_RETRY_EXIT_CODE }
if (-not $retryExitCode) { $retryExitCode = 17 }
$retryExitCode = [int]$retryExitCode

$env:RETRY_EXIT_CODE = $retryExitCode
$env:DEEP_CRAWL_RETRY_EXIT_CODE = $retryExitCode

# OUT_DIR from env or --out-dir
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

# Strict retry controls (retry phase only)
$strictMinSuccessRate = $env:STRICT_MIN_RETRY_SUCCESS_RATE
if (-not $strictMinSuccessRate) { $strictMinSuccessRate = 0.1 }
$strictMinSuccessRate = [double]$strictMinSuccessRate

$strictMaxRetryIter = $env:STRICT_MAX_RETRY_ITER
if (-not $strictMaxRetryIter) { $strictMaxRetryIter = 10 }
$strictMaxRetryIter = [int]$strictMaxRetryIter

# OOM auto restart controls (common SIGKILL exit code on Linux is 137)
$oomRestartLimit = $env:OOM_RESTART_LIMIT
if (-not $oomRestartLimit) { $oomRestartLimit = 100 }
$oomRestartLimit = [int]$oomRestartLimit

$oomRestartDelay = $env:OOM_RESTART_DELAY
if (-not $oomRestartDelay) { $oomRestartDelay = 10 }
$oomRestartDelay = [int]$oomRestartDelay

$oomRestartCount = 0

# History file (JSONL)
$retryHistoryFile = if ($env:RETRY_HISTORY_FILE) { $env:RETRY_HISTORY_FILE } else { (Join-Path $outDir "retry_history.jsonl") }
$historyDir = Split-Path $retryHistoryFile -Parent
if (-not (Test-Path $historyDir)) { New-Item -ItemType Directory -Force -Path $historyDir | Out-Null }

function Write-RetryHistory {
  param(
    [int]$Iteration,
    [int]$ExitCode,
    [int]$CurrentRetryCount,
    [int]$PrevRetryCount,
    [string]$Reason
  )
  $obj = [pscustomobject]@{
    timestamp           = (Get-Date).ToString("o")
    iteration           = $Iteration
    exit_code           = $ExitCode
    current_retry_count = $CurrentRetryCount
    prev_retry_count    = $PrevRetryCount
    reason              = $Reason
  }
  Add-Content -Path $retryHistoryFile -Value ($obj | ConvertTo-Json -Compress)
}

function Read-RetryFields {
  # returns PSCustomObject { retry_count, total_companies, attempted_total, all_attempted }
  $result = [pscustomobject]@{ retry_count = 0; total_companies = 0; attempted_total = 0; all_attempted = $false }
  if (-not (Test-Path $retryFile)) { return $result }

  try {
    $json = Get-Content $retryFile -Raw | ConvertFrom-Json
  } catch {
    return $result
  }

  if ($null -ne $json.retry_companies) {
    $result.retry_count = ($json.retry_companies | Measure-Object).Count
  }
  if ($json.PSObject.Properties.Name -contains "total_companies") {
    $result.total_companies = [int]$json.total_companies
  }
  if ($json.PSObject.Properties.Name -contains "attempted_total") {
    $result.attempted_total = [int]$json.attempted_total
  }
  if ($json.PSObject.Properties.Name -contains "all_attempted") {
    $result.all_attempted = [bool]$json.all_attempted
  }
  return $result
}

# Detect user supplied --retry-mode once; wrapper won't override if present
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

$prevRetryCount = 0
$iter = 1
$phase = "primary"     # primary => skip-retry (after first); retry => only-retry
$strictRetryCount = 0

while ($true) {
  Write-Host "[retry-wrapper] iteration $iter (phase=$phase)"

  # Decide retry-mode for this iteration (only if wrapper controls it)
  $retryCompanyMode = $null
  if (-not $userRetryModeProvided) {
    if ($phase -eq "retry") {
      $retryCompanyMode = "only-retry"
    } else {
      # primary
      if ($iter -eq 1) {
        # First run: use all only if retry file missing OR retry_companies empty
        if (-not (Test-Path $retryFile)) {
          $retryCompanyMode = "all"
        } else {
          $fields = Read-RetryFields
          if ([int]$fields.retry_count -eq 0) { $retryCompanyMode = "all" } else { $retryCompanyMode = "skip-retry" }
        }
      } else {
        # Later runs: always skip-retry in primary
        $retryCompanyMode = "skip-retry"
      }
    }
    Write-Host "[retry-wrapper] RETRY_COMPANY_MODE=$retryCompanyMode"
  } else {
    Write-Host "[retry-wrapper] using user-specified --retry-mode=$userRetryModeValue"
  }

  # Build args
  $cmdArgs = @()
  if (-not $userRetryModeProvided) { $cmdArgs += @("--retry-mode", $retryCompanyMode) }
  if ($RunArgs) { $cmdArgs += $RunArgs }

  & python "scripts/run.py" @cmdArgs
  $exitCode = [int]$LASTEXITCODE

  # ---------------- Clean exit ----------------
  if ($exitCode -eq 0) {
    $fields = Read-RetryFields
    Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount 0 -PrevRetryCount $prevRetryCount -Reason "clean_exit"

    # Defensive: if exit=0 but retry list exists, enter retry phase once.
    if ($phase -eq "primary" -and [int]$fields.retry_count -gt 0) {
      Write-Host "[retry-wrapper] run exited cleanly but retry_companies.json has $($fields.retry_count); switching to retry phase."
      $phase = "retry"
      $prevRetryCount = [int]$fields.retry_count
      $iter += 1
      continue
    }

    Write-Host "[retry-wrapper] run finished cleanly; stopping."
    break
  }

  # ---------------- Non-retry exit (OOM restart for 137 / 9) ----------------
  if ($exitCode -ne $retryExitCode) {
    if ($exitCode -eq 137 -or $exitCode -eq 9) {
      if ($oomRestartCount -lt $oomRestartLimit) {
        $oomRestartCount += 1
        Write-Host "[retry-wrapper] terminated (exit=$exitCode, likely OOM); restart $oomRestartCount/$oomRestartLimit after ${oomRestartDelay}s."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -Reason "oom_restart"
        Start-Sleep -Seconds $oomRestartDelay
        $iter += 1
        continue
      } else {
        Write-Host "[retry-wrapper] reached OOM_RESTART_LIMIT=$oomRestartLimit; stopping."
        Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -Reason "oom_stop"
        exit $exitCode
      }
    }

    Write-Host "[retry-wrapper] run exited with non-retry code $exitCode; stopping."
    Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -Reason "non_retry_exit"
    exit $exitCode
  }

  # ---------------- RETRY_EXIT_CODE path ----------------
  if (-not (Test-Path $retryFile)) {
    Write-Host "[retry-wrapper] retry exit code but $retryFile not found; stopping."
    Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount -1 -PrevRetryCount $prevRetryCount -Reason "retry_exit_missing_retry_file"
    exit 1
  }

  $fields = Read-RetryFields
  $currentRetryCount = [int]$fields.retry_count
  $totalCompanies = [int]$fields.total_companies
  $attemptedTotal = [int]$fields.attempted_total
  $allAttempted = [bool]$fields.all_attempted

  Write-Host "[retry-wrapper] retry_count=$currentRetryCount total_companies=$totalCompanies attempted_total=$attemptedTotal all_attempted=$allAttempted"

  $reason = "retry_exit_continue"

  # ---- Primary phase: keep doing skip-retry until everything attempted at least once ----
  if ($phase -eq "primary") {
    if ($allAttempted -or ($totalCompanies -gt 0 -and $attemptedTotal -ge $totalCompanies)) {
      Write-Host "[retry-wrapper] all primary companies attempted at least once; switching to retry phase (only-retry)."
      $phase = "retry"
      $strictRetryCount = 0
      $prevRetryCount = $currentRetryCount
    } else {
      Write-Host "[retry-wrapper] primary not finished yet ($attemptedTotal/$totalCompanies); staying in primary phase."
      $prevRetryCount = $currentRetryCount
    }

    Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount $currentRetryCount -PrevRetryCount $prevRetryCount -Reason $reason
    $iter += 1
    continue
  }

  # ---- Retry phase: enforce improvement + max iterations ----
  if ($prevRetryCount -eq 0) { $prevRetryCount = $currentRetryCount }

  if ($prevRetryCount -gt 0) {
    $succeeded = $prevRetryCount - $currentRetryCount
    $rate = 0.0
    if ($prevRetryCount -gt 0) { $rate = $succeeded / [double]$prevRetryCount }
    Write-Host ("[retry-wrapper] improvement: {0}/{1} ({2:P1})" -f $succeeded, $prevRetryCount, $rate)

    if ($currentRetryCount -eq 0) {
      Write-Host "[retry-wrapper] no companies left to retry; stopping."
      $reason = "retry_exit_stop_empty"
    } elseif ($rate -lt $strictMinSuccessRate) {
      Write-Host "[retry-wrapper] improvement below STRICT_MIN_RETRY_SUCCESS_RATE=$strictMinSuccessRate; stopping."
      $reason = "retry_exit_stop_progress"
    }
  }

  if ($reason -eq "retry_exit_continue") {
    if ($strictRetryCount -ge $strictMaxRetryIter) {
      Write-Host "[retry-wrapper] reached STRICT_MAX_RETRY_ITER=$strictMaxRetryIter; stopping."
      $reason = "retry_exit_stop_max_iter"
    }
  }

  Write-RetryHistory -Iteration $iter -ExitCode $exitCode -CurrentRetryCount $currentRetryCount -PrevRetryCount $prevRetryCount -Reason $reason

  if ($reason -ne "retry_exit_continue") { break }

  $prevRetryCount = $currentRetryCount
  $strictRetryCount += 1
  $iter += 1
}
