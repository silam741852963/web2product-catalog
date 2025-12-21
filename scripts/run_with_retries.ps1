<#
run_with_retries.ps1
PowerShell wrapper replicating the simplified bash logic:

- Runs: python3 scripts/run.py <args...>
- If exit code == 0:
  - If OUT_DIR/crawl_global_state.json has in_progress_companies > 0:
    - Run finalize pass: python3 scripts/run.py <args...> --finalize-in-progress-md --llm-mode none
  - Exit 0
- If exit code indicates SIGKILL (137):
  - Restart up to OOM_RESTART_LIMIT, waiting OOM_RESTART_DELAY seconds
- If exit code == RETRY_EXIT_CODE (default 17, env override):
  - Restart immediately
- Otherwise:
  - Exit with that code
#>

[CmdletBinding()]
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $Args
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Retry exit code: prefer RETRY_EXIT_CODE, then DEEP_CRAWL_RETRY_EXIT_CODE, else 17
$retryExitCode = $env:RETRY_EXIT_CODE
if ([string]::IsNullOrWhiteSpace($retryExitCode) -and -not [string]::IsNullOrWhiteSpace($env:DEEP_CRAWL_RETRY_EXIT_CODE)) {
  $retryExitCode = $env:DEEP_CRAWL_RETRY_EXIT_CODE
}
if ([string]::IsNullOrWhiteSpace($retryExitCode)) { $retryExitCode = "17" }

$env:RETRY_EXIT_CODE = $retryExitCode
$env:DEEP_CRAWL_RETRY_EXIT_CODE = $retryExitCode
[int]$retryExitCodeInt = 17
[void][int]::TryParse($retryExitCode, [ref]$retryExitCodeInt)

# --- Derive OUT_DIR from env or --out-dir argument (default: outputs)
$outDir = $env:OUT_DIR
if ([string]::IsNullOrWhiteSpace($outDir)) {
  $outDir = "outputs"
  for ($i = 0; $i -lt $Args.Count; $i++) {
    if ($Args[$i] -eq "--out-dir" -and ($i + 1) -lt $Args.Count) {
      $outDir = $Args[$i + 1]
      break
    }
  }
}

$globalStateFile = Join-Path $outDir "crawl_global_state.json"

# --- OOM auto restart config (SIGKILL -> exit 137)
[int]$oomRestartLimit = 100
[int]$oomRestartDelay = 10
if (-not [string]::IsNullOrWhiteSpace($env:OOM_RESTART_LIMIT)) { [void][int]::TryParse($env:OOM_RESTART_LIMIT, [ref]$oomRestartLimit) }
if (-not [string]::IsNullOrWhiteSpace($env:OOM_RESTART_DELAY)) { [void][int]::TryParse($env:OOM_RESTART_DELAY, [ref]$oomRestartDelay) }
[int]$oomRestartCount = 0

function Get-InProgressCount {
  param([string]$Path)

  if (-not (Test-Path -LiteralPath $Path)) { return 0 }

  try {
    $raw = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($raw)) { return 0 }

    $data = $raw | ConvertFrom-Json -ErrorAction Stop
    $lst = $data.in_progress_companies
    if ($null -eq $lst) { return 0 }

    # in case it's not an array for some reason
    if ($lst -is [System.Collections.IEnumerable] -and -not ($lst -is [string])) {
      return @($lst).Count
    }

    return 0
  } catch {
    return 0
  }
}

$iter = 1
while ($true) {
  Write-Host "[retry-wrapper] iteration=$iter out_dir=$outDir"

  & python3 scripts/run.py @Args
  $exitCode = $LASTEXITCODE

  # Clean exit
  if ($exitCode -eq 0) {
    $inprog = Get-InProgressCount -Path $globalStateFile
    if ($inprog -gt 0) {
      Write-Host "[retry-wrapper] clean exit but in_progress_companies=$inprog; running finalize pass (markdown only)."
      & python3 scripts/run.py @Args --finalize-in-progress-md --llm-mode none
      $finalizeExit = $LASTEXITCODE
      if ($finalizeExit -ne 0) {
        Write-Host "[retry-wrapper] finalize pass failed exit_code=$finalizeExit; stopping."
        exit $finalizeExit
      }
      Write-Host "[retry-wrapper] finalize pass completed."
    }

    Write-Host "[retry-wrapper] finished."
    exit 0
  }

  # OOM / SIGKILL handling (exit 128+9 = 137)
  if ($exitCode -ge 128) {
    $signal = $exitCode - 128
    if ($signal -eq 9) {
      if ($oomRestartCount -lt $oomRestartLimit) {
        $oomRestartCount++
        Write-Host "[retry-wrapper] killed by signal 9 (likely OOM); restart $oomRestartCount/$oomRestartLimit after ${oomRestartDelay}s."
        Start-Sleep -Seconds $oomRestartDelay
        $iter++
        continue
      } else {
        Write-Host "[retry-wrapper] repeated OOM; reached OOM_RESTART_LIMIT=$oomRestartLimit; stopping."
        exit $exitCode
      }
    }
  }

  # Retry requested by scheduler
  if ($exitCode -eq $retryExitCodeInt) {
    Write-Host "[retry-wrapper] retry exit_code=$exitCode; restarting."
    $iter++
    continue
  }

  # Anything else: stop
  Write-Host "[retry-wrapper] run exited with code=$exitCode; stopping."
  exit $exitCode
}
