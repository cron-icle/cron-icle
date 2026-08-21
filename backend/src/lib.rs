//! Chronicle: a local-first computer memory engine that runs as a single
//! standalone daemon, in the spirit of Postgres or Redis — one process owns
//! capture, persistence, local LLM inference, and an embedded HTTP server
//! that serves both a JSON API and the built frontend on `127.0.0.1`. There
//! is no installer, no code-signing/notarization step, and no native app
//! shell: `run()` binds a port, opens the user's browser to it, and keeps
//! running headless regardless of whether that tab stays open.

mod app_service;
mod config;
#[allow(dead_code)]
mod capture;
mod data_directory;
mod filesystem_watch;
mod hardware_profiler;
mod http;
#[allow(dead_code)]
mod inference;
mod memory_planner;
#[allow(dead_code)]
mod persistence;
#[allow(dead_code)]
mod processing;
mod state;

use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;
use state::AppState;
use std::sync::Arc;

#[derive(RustEmbed)]
#[folder = "../frontend/dist"]
struct FrontendAssets;

/// Serves the built React frontend from the binary's embedded assets (see
/// `FrontendAssets`, embedded from `frontend/dist/` at compile time). Falls
/// back to `index.html` for any path that isn't a known asset, so
/// client-side routing (if the frontend ever adds any) still works, matching
/// a typical single-page-app static-hosting setup.
async fn serve_frontend(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match FrontendAssets::get(path).or_else(|| FrontendAssets::get("index.html")) {
        Some(file) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            ([(header::CONTENT_TYPE, mime.as_ref())], file.data.into_owned()).into_response()
        }
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

async fn shutdown_signal(state: Arc<AppState>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(windows)]
    let windows_close = async {
        let mut close = tokio::signal::windows::ctrl_close().expect("failed to install ctrl-close handler");
        let mut shutdown = tokio::signal::windows::ctrl_shutdown().expect("failed to install ctrl-shutdown handler");
        tokio::select! {
            _ = close.recv() => {},
            _ = shutdown.recv() => {},
        }
    };
    #[cfg(not(windows))]
    let windows_close = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = windows_close => {},
    }
    tracing::info!("shutting down: stopping capture and local AI engine");
    app_service::stop_capture_state(&state);
    app_service::shutdown_llama_engine(&state);
}

pub async fn run() {
    tracing_subscriber::fmt()
        .with_env_filter("chronicle=info")
        .with_target(false)
        .init();

    // Never blocks: if the user hasn't chosen a data directory yet (see
    // `data_directory`), `AppState::initialize` below runs in a temporary,
    // non-persistent mode instead of prompting here. The user picks one
    // from Settings when they set up local AI.
    match data_directory::current() {
        Some(dir) => tracing::info!(path = %dir.display(), "using data directory"),
        None => tracing::info!("no data directory chosen yet; running in temporary mode until Settings is used to pick one"),
    }

    // Database open failures are recoverable: AppState::initialize() falls
    // back to a transient in-memory database and records the failure so the
    // UI can surface it, rather than panicking the whole process.
    let state = Arc::new(AppState::initialize());
    let capture_enabled = state
        .settings
        .lock()
        .map(|settings| settings.enabled)
        .unwrap_or(false);
    if capture_enabled {
        if let Err(error) = app_service::start_capture_state(&state) {
            tracing::error!(%error, "failed to start capture at launch");
        }
    }

    let app = axum::Router::new()
        .nest("/api", http::router())
        .fallback(serve_frontend)
        .with_state(state.clone())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let port = config::resolve_port();
    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, port, "failed to bind — is another Chronicle instance already running?");
            std::process::exit(1);
        }
    };

    let url = format!("http://127.0.0.1:{port}");
    tracing::info!(%url, "Chronicle is running");
    if config::should_open_browser() {
        if let Err(error) = open::that(&url) {
            tracing::warn!(%error, "failed to open the browser automatically");
        }
    }

    if let Err(error) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await
    {
        tracing::error!(%error, "Chronicle exited because the HTTP server failed");
        std::process::exit(1);
    }
}
