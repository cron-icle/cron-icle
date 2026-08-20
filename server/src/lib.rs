mod data_directory;
#[allow(dead_code)]
mod activity_capture;
#[allow(dead_code)]
mod asynchronous_processing_queue;
mod capture_writer;
#[allow(dead_code)]
mod embedding_provider;
mod filesystem_activity_capture;
mod http_api;
#[allow(dead_code)]
mod input_capture;
mod local_inference_setup;
mod hardware_profiler;
mod local_model_provider;
mod memory_planner;
mod native_inference;
#[allow(dead_code)]
mod local_semantic_processing;
mod local_sqlite_event_database;
mod tauri_application_commands;
#[allow(dead_code)]
mod transient_screenshot_capture;
mod windows_active_window_screenshot;
mod windows_graphics_capture_session;
#[allow(dead_code)]
mod windows_ui_automation_capture;

use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;
use std::sync::Arc;
use tauri_application_commands::AppState;

#[derive(RustEmbed)]
#[folder = "../dist"]
struct FrontendAssets;

const DEFAULT_PORT: u16 = 47823;

fn resolve_port() -> u16 {
    std::env::var("CHRONICLE_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

/// Serves the built React frontend from the binary's embedded assets (see
/// `FrontendAssets`, embedded from `dist/` at compile time). Falls back to
/// `index.html` for any path that isn't a known asset, so client-side
/// routing (if the frontend ever adds any) still works, matching a typical
/// single-page-app static-hosting setup.
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
    tauri_application_commands::stop_capture_state(&state);
    tauri_application_commands::shutdown_llama_engine(&state);
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
        if let Err(error) = tauri_application_commands::start_capture_state(&state) {
            tracing::error!(%error, "failed to start capture at launch");
        }
    }

    let app = axum::Router::new()
        .nest("/api", http_api::router())
        .fallback(serve_frontend)
        .with_state(state.clone())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let port = resolve_port();
    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, port, "failed to bind — is another Chronicle instance already running?");
            std::process::exit(1);
        }
    };

    let url = format!("http://127.0.0.1:{port}");
    tracing::info!(%url, "Chronicle is running");
    if std::env::var("CHRONICLE_NO_OPEN").is_err() {
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
