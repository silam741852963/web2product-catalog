$python = "python"

# Pass script and arguments as an array so PowerShell doesn't treat it as one big string
$args = @(
    "scripts/run.py",
    "--company-file", "data/test/test_20.csv",
    "--log-level", "DEBUG",
    "--company-concurrency", "16",
    "--enable-resource-monitor",
    "--max-pages", "5"
)

while ($true) {
    & $python @args
    $code = $LASTEXITCODE
    Write-Host "[wrapper] run.py exited with code $code"

    if ($code -eq 0) {
        Write-Host "[wrapper] Normal exit (0) - not restarting."
        break
    }

    if ($code -eq 3) {
        Write-Host "[wrapper] Stall (3) - restarting in 10 seconds..."
        Start-Sleep -Seconds 10
        continue
    }

    # Either restart on all non-zero, or stop - your choice:
    Write-Host "[wrapper] Non-zero exit ($code) - restarting in 10 seconds..."
    Start-Sleep -Seconds 10
}