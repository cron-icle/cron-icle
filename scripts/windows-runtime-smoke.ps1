param(
    [string]$Executable = "$PSScriptRoot\..\backend\target\release\cronicle.exe",
    [int]$StartupTimeoutSeconds = 15
)

$ErrorActionPreference = "Stop"

$resolvedExecutable = (Resolve-Path -LiteralPath $Executable -ErrorAction Stop).Path
Write-Host "Starting Cronicle runtime: $resolvedExecutable"
$process = Start-Process -FilePath $resolvedExecutable -PassThru
try {
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    do {
        Start-Sleep -Milliseconds 250
        $process.Refresh()
        if ($process.HasExited) {
            throw "Cronicle exited during startup with code $($process.ExitCode)"
        }
    } while ((Get-Date) -lt $deadline)

    Write-Host "Cronicle remained running for $StartupTimeoutSeconds seconds."
}
finally {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    $process.Dispose()
}

Write-Host "Windows runtime smoke test passed."
