param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)

$retryExitCode   = $env:RETRY_EXIT_CODE
if (-not $retryExitCode)   { $retryExitCode = 17 }

$maxRetryIter    = $env:MAX_RETRY_ITER
if (-not $maxRetryIter)    { $maxRetryIter = 10 }

$minSuccessRate  = $env:MIN_RETRY_SUCCESS_RATE
if (-not $minSuccessRate)  { $minSuccessRate = 0.3 }

$outDir          = $env:OUT_DIR
if (-not $outDir)          { $outDir = "outputs" }

$prevRetryCount = 0
$iter = 1

while ($true) {
    Write-Host "[retry-wrapper] iteration ${iter}: running crawler..."
    & python scripts/run.py @RunArgs
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host "[retry-wrapper] run finished cleanly (exit 0); stopping."
        break
    }

    if ($exitCode -ne [int]$retryExitCode) {
        Write-Host "[retry-wrapper] run exited with non-retry code $exitCode; stopping."
        exit $exitCode
    }

    $retryFile = Join-Path $outDir "retry_companies.json"
    if (-not (Test-Path $retryFile)) {
        Write-Host "[retry-wrapper] retry exit code but $retryFile not found; stopping."
        exit 1
    }

    $json = Get-Content $retryFile -Raw | ConvertFrom-Json
    $currentRetryCount = ($json.retry_companies | Measure-Object).Count
    Write-Host "[retry-wrapper] $currentRetryCount companies need retry."

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
            break
        }

        if ($rate -lt [double]$minSuccessRate) {
            Write-Host "[retry-wrapper] progress below MIN_RETRY_SUCCESS_RATE=$minSuccessRate; stop auto retries."
            break
        }
    }

    if ($iter -ge [int]$maxRetryIter) {
        Write-Host "[retry-wrapper] reached MAX_RETRY_ITER=$maxRetryIter; stopping."
        break
    }

    $prevRetryCount = $currentRetryCount
    $iter += 1
}