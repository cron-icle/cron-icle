$ErrorActionPreference = "Stop"

function Measure-Step([string]$Name, [scriptblock]$Action) {
    Write-Host "`n== $Name =="
    $elapsed = Measure-Command { & $Action }
    if ($LASTEXITCODE -ne 0) { throw "$Name failed" }
    Write-Host ("Elapsed: {0:N2}s" -f $elapsed.TotalSeconds)
}

Measure-Step "raw persistence baseline" {
    cargo test --manifest-path backend/Cargo.toml --lib --offline persists_one_thousand_events -- --nocapture
}

Measure-Step "semantic search baseline" {
    cargo test --manifest-path backend/Cargo.toml --lib --offline fts_search_has_bounded_latency_at_one_thousand_events -- --nocapture
}

Measure-Step "queue throughput baseline" {
    cargo test --manifest-path backend/Cargo.toml --lib --offline busy_worker_processes_bounded_work_and_stops -- --nocapture
}

Measure-Step "frontend production build" {
    npm run build
}

Write-Host "`nBenchmark workflow completed. Native screenshot and restricted-app measurements require the Windows capture acceptance environment."
