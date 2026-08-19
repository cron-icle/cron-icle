//! HTTP surface for Chronicle's daemon binary.
//!
//! Chronicle runs as a single long-lived process (no installer, no native
//! app shell) that exposes its state over a local-only JSON API on
//! `127.0.0.1`. This module is a thin translation layer: every handler here
//! extracts the request, forwards to a plain function in
//! `tauri_application_commands`/`local_inference_setup` (the actual business
//! logic, unaware of HTTP), and serializes the result. See `lib.rs` for
//! where this router is mounted alongside the static frontend assets.

use crate::local_inference_setup;
use crate::local_sqlite_event_database::{RawEvent, SemanticEvent};
use crate::tauri_application_commands::{self as commands, AppState, CaptureStatus};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

type SharedState = Arc<AppState>;

/// Wraps the `Result<T, String>` shape every business-logic function already
/// returns into an HTTP response: `Ok` serializes as `200 application/json`,
/// `Err` becomes `500` with `{"error": ...}`.
struct AppError(String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": self.0 }))).into_response()
    }
}

impl From<String> for AppError {
    fn from(value: String) -> Self {
        AppError(value)
    }
}

type ApiResult<T> = Result<Json<T>, AppError>;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/health", get(health_check))
        .route("/screenshot/active-window", post(capture_active_window_screenshot))
        .route(
            "/screenshot/graphics-capture-available",
            get(graphics_capture_session_available),
        )
        .route("/events/recent-count", get(recent_event_count))
        .route("/events", get(list_events).post(record_event))
        .route("/events/semantic", get(list_semantic_events).post(record_semantic_event))
        .route("/events/raw-overview", get(list_raw_event_processing_overview))
        .route("/events/{id}/semantic", get(semantic_for_event))
        .route("/data/delete-all", post(delete_all_data))
        .route("/data/export", get(export_data))
        .route("/startup-diagnostics", get(startup_diagnostics))
        .route("/settings/capture", get(get_capture_settings).put(update_capture_settings))
        .route("/settings/input-permission", put(set_input_permission))
        .route("/settings/screenshot-permission", put(set_screenshot_permission))
        .route("/settings/keyboard-text-allowlist", put(set_keyboard_text_allowlist))
        .route("/settings/excluded-applications", put(set_excluded_applications))
        .route("/settings/excluded-paths", put(set_excluded_paths))
        .route("/settings/watched-folders", put(set_watched_folders))
        .route("/capture/start", post(start_capture))
        .route("/capture/stop", post(stop_capture))
        .route("/capture/status", get(capture_status))
        .route("/processing/queue-status", get(processing_queue_status))
        .route("/processing/metrics", get(processing_metrics))
        .route("/processing/queue-limits", get(processing_queue_limits))
        .route("/processing/cancel-pending", post(cancel_pending_processing_tasks))
        .route("/processing/retry-failed", post(retry_failed_processing_tasks))
        .route("/processing/status/{event_id}", get(processing_status_for_event))
        .route("/inference/telemetry", get(inference_telemetry))
        .route("/storage/usage", get(storage_usage))
        .route("/model-provider/status", get(model_provider_status))
        .route("/diagnostics/capture", get(capture_diagnostics))
        .route("/data-directory", get(get_data_directory))
        .route("/data-directory/change", post(change_data_directory))
        .route("/data-directory/move-progress", get(data_directory_move_progress))
        .route("/local-ai/setup-status", get(local_ai_setup_status))
        .route("/local-ai/cancel-download", post(cancel_model_download))
        .route("/local-ai/download-chat-model", post(setup_download_chat_model))
        .route("/local-ai/download-embed-model", post(setup_download_embed_model))
        .route("/local-ai/start-engine", post(setup_start_engine))
        .route("/local-ai/chat-model", delete(setup_remove_chat_model))
        .route("/local-ai/embed-model", delete(setup_remove_embed_model))
        .route("/local-ai/progress", get(local_ai_download_progress))
}

async fn health_check() -> &'static str {
    commands::health_check()
}

#[derive(Deserialize)]
struct WindowHandleBody {
    window_handle: isize,
}

async fn capture_active_window_screenshot(Json(body): Json<WindowHandleBody>) -> Result<Response, AppError> {
    let bytes = commands::capture_active_window_screenshot(body.window_handle)?;
    Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response())
}

#[derive(Deserialize)]
struct WindowHandleQuery {
    window_handle: isize,
}

async fn graphics_capture_session_available(Query(query): Query<WindowHandleQuery>) -> ApiResult<bool> {
    Ok(Json(commands::graphics_capture_session_available(query.window_handle)?))
}

async fn recent_event_count(State(state): State<SharedState>) -> ApiResult<i64> {
    Ok(Json(commands::recent_event_count(&state).await?))
}

#[derive(Deserialize)]
struct ListQuery {
    limit: u32,
    query: Option<String>,
}

async fn list_events(State(state): State<SharedState>, Query(query): Query<ListQuery>) -> ApiResult<Vec<RawEvent>> {
    Ok(Json(commands::list_events(&state, query.limit, query.query).await?))
}

async fn record_event(State(state): State<SharedState>, Json(event): Json<RawEvent>) -> Result<StatusCode, AppError> {
    commands::record_event(&state, event)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_semantic_events(
    State(state): State<SharedState>,
    Query(query): Query<ListQuery>,
) -> ApiResult<Vec<crate::local_sqlite_event_database::SemanticEventView>> {
    Ok(Json(commands::list_semantic_events(&state, query.limit, query.query).await?))
}

async fn record_semantic_event(
    State(state): State<SharedState>,
    Json(event): Json<SemanticEvent>,
) -> Result<StatusCode, AppError> {
    commands::record_semantic_event(&state, event)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct OverviewQuery {
    limit: u32,
}

async fn list_raw_event_processing_overview(
    State(state): State<SharedState>,
    Query(query): Query<OverviewQuery>,
) -> ApiResult<Vec<commands::RawEventProcessingOverview>> {
    Ok(Json(commands::list_raw_event_processing_overview(&state, query.limit).await?))
}

async fn semantic_for_event(
    State(state): State<SharedState>,
    AxumPath(id): AxumPath<String>,
) -> ApiResult<Option<SemanticEvent>> {
    Ok(Json(commands::semantic_for_event(&state, id).await?))
}

async fn delete_all_data(State(state): State<SharedState>) -> Result<StatusCode, AppError> {
    commands::delete_all_data(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn export_data(State(state): State<SharedState>) -> ApiResult<String> {
    Ok(Json(commands::export_data(&state).await?))
}

async fn startup_diagnostics(State(state): State<SharedState>) -> Json<Option<String>> {
    Json(commands::startup_diagnostics(&state))
}

async fn get_capture_settings(
    State(state): State<SharedState>,
) -> ApiResult<crate::activity_capture::CaptureSettings> {
    Ok(Json(commands::get_capture_settings(&state)?))
}

async fn update_capture_settings(
    State(state): State<SharedState>,
    Json(settings): Json<crate::activity_capture::CaptureSettings>,
) -> ApiResult<crate::activity_capture::CaptureSettings> {
    Ok(Json(commands::update_capture_settings(&state, settings)?))
}

#[derive(Deserialize)]
struct InputPermissionBody {
    input: String,
    enabled: bool,
}

async fn set_input_permission(
    State(state): State<SharedState>,
    Json(body): Json<InputPermissionBody>,
) -> ApiResult<crate::activity_capture::CaptureSettings> {
    Ok(Json(commands::set_input_permission(&state, body.input, body.enabled)?))
}

#[derive(Deserialize)]
struct EnabledBody {
    enabled: bool,
}

async fn set_screenshot_permission(
    State(state): State<SharedState>,
    Json(body): Json<EnabledBody>,
) -> ApiResult<crate::activity_capture::CaptureSettings> {
    Ok(Json(commands::set_screenshot_permission(&state, body.enabled)?))
}

#[derive(Deserialize)]
struct ApplicationsBody {
    applications: Vec<String>,
}

async fn set_keyboard_text_allowlist(
    State(state): State<SharedState>,
    Json(body): Json<ApplicationsBody>,
) -> ApiResult<crate::activity_capture::CaptureSettings> {
    Ok(Json(commands::set_keyboard_text_allowlist(&state, body.applications)?))
}

async fn set_excluded_applications(
    State(state): State<SharedState>,
    Json(body): Json<ApplicationsBody>,
) -> ApiResult<crate::activity_capture::CaptureSettings> {
    Ok(Json(commands::set_excluded_applications(&state, body.applications)?))
}

#[derive(Deserialize)]
struct PathsBody {
    paths: Vec<String>,
}

async fn set_excluded_paths(
    State(state): State<SharedState>,
    Json(body): Json<PathsBody>,
) -> ApiResult<crate::activity_capture::CaptureSettings> {
    Ok(Json(commands::set_excluded_paths(&state, body.paths)?))
}

#[derive(Deserialize)]
struct FoldersBody {
    folders: Vec<String>,
}

async fn set_watched_folders(
    State(state): State<SharedState>,
    Json(body): Json<FoldersBody>,
) -> ApiResult<crate::activity_capture::CaptureSettings> {
    Ok(Json(commands::set_watched_folders(&state, body.folders)?))
}

async fn start_capture(State(state): State<SharedState>) -> Result<StatusCode, AppError> {
    commands::start_capture(&state)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn stop_capture(State(state): State<SharedState>) -> Result<StatusCode, AppError> {
    commands::stop_capture(&state)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn capture_status(State(state): State<SharedState>) -> ApiResult<CaptureStatus> {
    Ok(Json(commands::capture_status(&state).await?))
}

async fn processing_queue_status(State(state): State<SharedState>) -> ApiResult<commands::ProcessingQueueStatus> {
    Ok(Json(commands::processing_queue_status(&state).await?))
}

async fn processing_metrics(State(state): State<SharedState>) -> ApiResult<commands::ProcessingMetricsView> {
    Ok(Json(commands::processing_metrics(&state)?))
}

async fn processing_queue_limits() -> Json<commands::ProcessingQueueLimits> {
    Json(commands::processing_queue_limits())
}

async fn cancel_pending_processing_tasks(State(state): State<SharedState>) -> ApiResult<usize> {
    Ok(Json(commands::cancel_pending_processing_tasks(&state).await?))
}

async fn retry_failed_processing_tasks(State(state): State<SharedState>) -> ApiResult<usize> {
    Ok(Json(commands::retry_failed_processing_tasks(&state).await?))
}

async fn processing_status_for_event(
    State(state): State<SharedState>,
    AxumPath(raw_event_id): AxumPath<String>,
) -> ApiResult<Vec<commands::EventProcessingStatus>> {
    Ok(Json(commands::processing_status_for_event(&state, raw_event_id).await?))
}

async fn inference_telemetry() -> Json<commands::InferenceTelemetryView> {
    Json(commands::inference_telemetry())
}

async fn storage_usage(State(state): State<SharedState>) -> ApiResult<std::collections::HashMap<String, i64>> {
    Ok(Json(commands::storage_usage(&state).await?))
}

async fn model_provider_status() -> ApiResult<commands::ModelProviderStatus> {
    Ok(Json(commands::model_provider_status().await?))
}

async fn capture_diagnostics(State(state): State<SharedState>) -> ApiResult<commands::CaptureDiagnostics> {
    Ok(Json(commands::capture_diagnostics(&state).await?))
}

async fn get_data_directory() -> Json<Option<String>> {
    Json(commands::get_data_directory())
}

#[derive(Serialize)]
struct Restarting {
    restarting: bool,
}

/// Runs the data-directory move synchronously (the browser tab watches
/// `/data-directory/move-progress` while this is in flight), then responds
/// `200` and relaunches the process shortly after — delayed just long
/// enough for this response to flush before the process exits.
async fn change_data_directory(State(state): State<SharedState>) -> ApiResult<Restarting> {
    commands::change_data_directory(state.clone()).await?;
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        if let Err(error) = commands::relaunch_self() {
            tracing::error!(%error, "failed to relaunch after data directory change");
        }
        std::process::exit(0);
    });
    Ok(Json(Restarting { restarting: true }))
}

async fn data_directory_move_progress(State(state): State<SharedState>) -> impl IntoResponse {
    match commands::data_directory_move_progress(&state) {
        Some(progress) => Json(progress).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}

async fn local_ai_setup_status() -> Json<local_inference_setup::LlamaSetupStatus> {
    Json(local_inference_setup::local_ai_setup_status().await)
}

async fn cancel_model_download(State(state): State<SharedState>) -> Result<StatusCode, AppError> {
    local_inference_setup::cancel_model_download(&state)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Downloads are fire-and-forget from the HTTP caller's perspective: this
/// returns `202` immediately and the browser polls `/local-ai/progress` (and
/// `/local-ai/setup-status`) to track completion, since a multi-hundred-MB
/// download would otherwise hold the request open for minutes.
async fn setup_download_chat_model(State(state): State<SharedState>) -> StatusCode {
    tokio::spawn(async move {
        if let Err(error) = local_inference_setup::setup_download_chat_model(state).await {
            tracing::error!(%error, "chat model download failed");
        }
    });
    StatusCode::ACCEPTED
}

async fn setup_download_embed_model(State(state): State<SharedState>) -> StatusCode {
    tokio::spawn(async move {
        if let Err(error) = local_inference_setup::setup_download_embed_model(state).await {
            tracing::error!(%error, "embedding model download failed");
        }
    });
    StatusCode::ACCEPTED
}

async fn setup_start_engine(State(state): State<SharedState>) -> Result<StatusCode, AppError> {
    local_inference_setup::setup_start_engine(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn setup_remove_chat_model(State(state): State<SharedState>) -> Result<StatusCode, AppError> {
    local_inference_setup::setup_remove_chat_model(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn setup_remove_embed_model(State(state): State<SharedState>) -> Result<StatusCode, AppError> {
    local_inference_setup::setup_remove_embed_model(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn local_ai_download_progress(State(state): State<SharedState>) -> impl IntoResponse {
    match local_inference_setup::download_progress(&state) {
        Some(progress) => Json(progress).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}
