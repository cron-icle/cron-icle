$ErrorActionPreference = "Stop"

Write-Host "Running frontend checks..."
npm run test:frontend
if ($LASTEXITCODE -ne 0) { throw "Frontend checks failed" }

Write-Host "Running Rust tests..."
cargo test --manifest-path backend/Cargo.toml --lib --offline
if ($LASTEXITCODE -ne 0) { throw "Rust tests failed" }

Write-Host "Building frontend..."
npm run build
if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }

Write-Host "Building NSIS installer..."
npm run tauri build -- --bundles nsis
if ($LASTEXITCODE -ne 0) { throw "NSIS build failed" }

Write-Host "Starting runtime smoke test..."
& "$PSScriptRoot/windows-runtime-smoke.ps1"
if ($LASTEXITCODE -ne 0) { throw "Runtime smoke test failed" }

Write-Host "Release smoke test passed."
