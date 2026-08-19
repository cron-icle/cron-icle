<#
.SYNOPSIS
    Installs the latest (or a specific) Chronicle release binary.

.DESCRIPTION
    Chronicle ships as a single unsigned binary - no installer, no
    code-signing/notarization (see the README's Installation section for
    why). This script downloads the release asset from GitHub Releases,
    places it under %LOCALAPPDATA%\Programs\Chronicle, and adds that folder
    to the current user's PATH so `chronicle` works from any terminal.

    Typical usage (curl, from any terminal):
        curl.exe -sSL -o "$env:TEMP\chronicle-install.ps1" https://raw.githubusercontent.com/anadi45/chronicle/main/scripts/install.ps1; powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\chronicle-install.ps1"

    NOTE: this deliberately downloads to a file and runs it with -File
    rather than piping straight into `powershell -Command -`. A multi-line
    param() block (this one included) silently no-ops when fed to
    `-Command -` via a pipe on Windows PowerShell 5.1 (it parses the stream
    incrementally rather than as a whole script) — download-then-run is the
    form that's actually verified to work end to end.

    To install a specific version instead of latest, pass -Version:
        powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\chronicle-install.ps1" -Version v1.2.3
#>
param([string]$Version = "latest", [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Programs\Chronicle"))
# InstallDir deliberately isn't %LOCALAPPDATA%\Chronicle: that's a natural
# folder name for a user to also pick as Chronicle's *data* directory (event
# database, downloaded models) from Settings, and this installer must never
# write into that folder.

$ErrorActionPreference = "Stop"
$repo = "anadi45/chronicle"
$assetName = "chronicle-windows-x86_64.exe"

# Every network call below passes -ErrorAction Stop explicitly (belt and
# suspenders alongside the global $ErrorActionPreference), and the whole
# body runs inside try/catch, so a failure always aborts with a clear
# message and a non-zero exit code instead of continuing into a broken
# download or PATH edit.
function Get-ReleaseAsset {
    param([string]$Version)
    $uri = if ($Version -eq "latest") {
        "https://api.github.com/repos/$repo/releases/latest"
    } else {
        "https://api.github.com/repos/$repo/releases/tags/$Version"
    }
    Write-Host "Looking up $Version release of $repo..."
    $release = Invoke-RestMethod -Uri $uri -Headers @{ "User-Agent" = "chronicle-install-script" } -ErrorAction Stop
    $asset = $release.assets | Where-Object { $_.name -eq $assetName }
    if (-not $asset) {
        throw "Could not find an asset named '$assetName' on release '$($release.tag_name)'. Available assets: $($release.assets.name -join ', ')"
    }
    return @{ Url = $asset.browser_download_url; Tag = $release.tag_name }
}

try {
    $asset = Get-ReleaseAsset -Version $Version
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $exePath = Join-Path $InstallDir "chronicle.exe"

    Write-Host "Downloading Chronicle $($asset.Tag) to $exePath..."
    Invoke-WebRequest -Uri $asset.Url -OutFile $exePath -UseBasicParsing -ErrorAction Stop

    # Downloaded files carry Windows' quarantine-equivalent (Zone.Identifier);
    # clearing it avoids an extra "this file came from the internet" prompt
    # beyond the SmartScreen warning the README already calls out.
    Unblock-File -Path $exePath -ErrorAction SilentlyContinue

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (($userPath -split ";") -notcontains $InstallDir) {
        Write-Host "Adding $InstallDir to your user PATH (restart your terminal to pick it up)..."
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
        $env:Path = "$env:Path;$InstallDir"
    }

    Write-Host ""
    Write-Host "Chronicle $($asset.Tag) installed to $exePath" -ForegroundColor Green
    Write-Host "Run it with: chronicle"
    Write-Host "(First run: Windows SmartScreen may warn this is an unrecognized app - click 'More info' then 'Run anyway'. This is expected for an unsigned binary; see the README.)"
} catch {
    Write-Host ""
    Write-Host "Chronicle install failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
