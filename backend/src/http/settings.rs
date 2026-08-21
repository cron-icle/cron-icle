//! `/api/settings/*` routes.

use super::errors::ApiResult;
use crate::app_service as service;
use crate::capture::activity::CaptureSettings;
use crate::state::AppState;
use axum::extract::State;
use axum::routing::{get, put};
use axum::{Json, Router};
use serde::Deserialize;
use std::sync::Arc;

type SharedState = Arc<AppState>;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/settings/capture", get(get_capture_settings).put(update_capture_settings))
        .route("/settings/input-permission", put(set_input_permission))
        .route("/settings/screenshot-permission", put(set_screenshot_permission))
        .route("/settings/keyboard-text-allowlist", put(set_keyboard_text_allowlist))
        .route("/settings/excluded-applications", put(set_excluded_applications))
        .route("/settings/excluded-paths", put(set_excluded_paths))
        .route("/settings/watched-folders", put(set_watched_folders))
}

async fn get_capture_settings(State(state): State<SharedState>) -> ApiResult<CaptureSettings> {
    Ok(Json(service::get_capture_settings(&state)?))
}

async fn update_capture_settings(
    State(state): State<SharedState>,
    Json(settings): Json<CaptureSettings>,
) -> ApiResult<CaptureSettings> {
    Ok(Json(service::update_capture_settings(&state, settings)?))
}

#[derive(Deserialize)]
struct InputPermissionBody {
    input: String,
    enabled: bool,
}

async fn set_input_permission(
    State(state): State<SharedState>,
    Json(body): Json<InputPermissionBody>,
) -> ApiResult<CaptureSettings> {
    Ok(Json(service::set_input_permission(&state, body.input, body.enabled)?))
}

#[derive(Deserialize)]
struct EnabledBody {
    enabled: bool,
}

async fn set_screenshot_permission(
    State(state): State<SharedState>,
    Json(body): Json<EnabledBody>,
) -> ApiResult<CaptureSettings> {
    Ok(Json(service::set_screenshot_permission(&state, body.enabled)?))
}

#[derive(Deserialize)]
struct ApplicationsBody {
    applications: Vec<String>,
}

async fn set_keyboard_text_allowlist(
    State(state): State<SharedState>,
    Json(body): Json<ApplicationsBody>,
) -> ApiResult<CaptureSettings> {
    Ok(Json(service::set_keyboard_text_allowlist(&state, body.applications)?))
}

async fn set_excluded_applications(
    State(state): State<SharedState>,
    Json(body): Json<ApplicationsBody>,
) -> ApiResult<CaptureSettings> {
    Ok(Json(service::set_excluded_applications(&state, body.applications)?))
}

#[derive(Deserialize)]
struct PathsBody {
    paths: Vec<String>,
}

async fn set_excluded_paths(
    State(state): State<SharedState>,
    Json(body): Json<PathsBody>,
) -> ApiResult<CaptureSettings> {
    Ok(Json(service::set_excluded_paths(&state, body.paths)?))
}

#[derive(Deserialize)]
struct FoldersBody {
    folders: Vec<String>,
}

async fn set_watched_folders(
    State(state): State<SharedState>,
    Json(body): Json<FoldersBody>,
) -> ApiResult<CaptureSettings> {
    Ok(Json(service::set_watched_folders(&state, body.folders)?))
}
