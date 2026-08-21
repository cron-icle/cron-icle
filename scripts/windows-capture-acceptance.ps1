param(
    [string]$Executable = "$PSScriptRoot\..\backend\target\release\cronicle.exe",
    [string]$Database = "$PSScriptRoot\..\cronicle.db",
    [int]$ObservationSeconds = 8
)

$ErrorActionPreference = "Stop"
$resolvedExecutable = (Resolve-Path -LiteralPath $Executable -ErrorAction Stop).Path
$resolvedDatabase = if ([System.IO.Path]::IsPathRooted($Database)) {
    [System.IO.Path]::GetFullPath($Database)
} else {
    [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Database))
}
$startedAt = (Get-Date).ToUniversalTime().ToString("o")
$cronicle = Start-Process -FilePath $resolvedExecutable -WorkingDirectory (Split-Path $resolvedDatabase) -PassThru
$foreground = Start-Process -FilePath "notepad.exe" -PassThru
try {
    Start-Sleep -Seconds $ObservationSeconds
    $cronicle.Refresh()
    if ($cronicle.HasExited) { throw "Cronicle exited with code $($cronicle.ExitCode)" }
    if (-not (Test-Path -LiteralPath $resolvedDatabase)) { throw "Cronicle did not create $resolvedDatabase" }

    $python = @'
import sqlite3, sys
db, started_at = sys.argv[1], sys.argv[2]
connection = sqlite3.connect(db)
row = connection.execute("SELECT COUNT(*) FROM raw_events WHERE source = 'foreground_window' AND created_at >= ?", (started_at,)).fetchone()
print(row[0])
'@
    $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { $pythonCommand = Get-Command python -ErrorAction SilentlyContinue }
    if (-not $pythonCommand -or $pythonCommand.Source -like "*WindowsApps*") {
        throw "Python 3 is required for the SQLite assertion; install Python or add it to PATH."
    }
    $count = ($python | & $pythonCommand.Source - $resolvedDatabase $before).Trim()
    if ([int]$count -eq 0) {
        Write-Warning "No foreground events were recorded. Verify Capture is enabled in Settings and rerun."
        exit 2
    }
    Write-Host "Windows capture acceptance passed: $count foreground events found."
}
finally {
    if (-not $foreground.HasExited) { Stop-Process -Id $foreground.Id -Force }
    if (-not $cronicle.HasExited) { Stop-Process -Id $cronicle.Id -Force }
    $foreground.Dispose(); $cronicle.Dispose()
}
