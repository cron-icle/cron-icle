#requires -version 5.1
<#
.SYNOPSIS
    Installs the latest (or a specific) Chronicle release binary.

.DESCRIPTION
    Chronicle ships as a single unsigned binary - no installer, no
    code-signing/notarization (see the README's Installation section for
    why). This script downloads the release asset from GitHub Releases,
    places it under %LOCALAPPDATA%\Chronicle, and adds that folder to the
    current user's PATH so `chronicle` works from any terminal.

    Typical usage (from PowerShell):
        irm https://raw.githubusercontent.com/anadi45/chronicle/main/scripts/install.ps1 | iex

    To install a specific version instead of latest, download this script
    first and pass -Version:
        .\install.ps1 -Version v1.2.3
#>
param(
    [string]$Version = "latest",
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Chronicle")
)

$ErrorActionPreference = "Stop"
$repo = "anadi45/chronicle"
$assetName = "chronicle-windows-x86_64.exe"

function Get-ReleaseAsset {
    param([string]$Version)
    $uri = if ($Version -eq "latest") {
        "https://api.github.com/repos/$repo/releases/latest"
    } else {
        "https://api.github.com/repos/$repo/releases/tags/$Version"
    }
    Write-Host "Looking up $Version release of $repo..."
    $release = Invoke-RestMethod -Uri $uri -Headers @{ "User-Agent" = "chronicle-install-script" }
    $asset = $release.assets | Where-Object { $_.name -eq $assetName }
    if (-not $asset) {
        throw "Could not find an asset named '$assetName' on release '$($release.tag_name)'. Available assets: $($release.assets.name -join ', ')"
    }
    return @{ Url = $asset.browser_download_url; Tag = $release.tag_name }
}

$asset = Get-ReleaseAsset -Version $Version
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$exePath = Join-Path $InstallDir "chronicle.exe"

Write-Host "Downloading Chronicle $($asset.Tag) to $exePath..."
Invoke-WebRequest -Uri $asset.Url -OutFile $exePath -UseBasicParsing

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
