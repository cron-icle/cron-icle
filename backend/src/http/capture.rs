//! `/api/capture/*` and `/api/screenshot/*` routes.

use super::errors::{ApiError, ApiResult};
use crate::app_service::{self as service, CaptureStatus};
use crate::state::AppState;
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::sync::Arc;

type SharedState = Arc<AppState>;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/screenshot/active-window", post(capture_active_window_screenshot))
        .route(
            "/screenshot/graphics-capture-available",
            get(graphics_capture_session_available),
        )
        .route("/capture/start", post(start_capture))
        .route("/capture/stop", post(stop_capture))
        .route("/capture/status", get(capture_status))
}

#[derive(Deserialize)]
struct WindowHandleBody {
    window_handle: isize,
}

async fn capture_active_window_screenshot(Json(body): Json<WindowHandleBody>) -> Result<Response, ApiError> {
    let bytes = service::capture_active_window_screenshot(body.window_handle)?;
    Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response())
}

#[derive(Deserialize)]
struct WindowHandleQuery {
    window_handle: isize,
}

async fn graphics_capture_session_available(Query(query): Query<WindowHandleQuery>) -> ApiResult<bool> {
    Ok(Json(service::graphics_capture_session_available(query.window_handle)?))
}

async fn start_capture(State(state): State<SharedState>) -> Result<StatusCode, ApiError> {
    service::start_capture(&state)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn stop_capture(State(state): State<SharedState>) -> Result<StatusCode, ApiError> {
    service::stop_capture(&state)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn capture_status(State(state): State<SharedState>) -> ApiResult<CaptureStatus> {
    Ok(Json(service::capture_status(&state).await?))
}
