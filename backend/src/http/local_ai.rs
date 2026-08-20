//! `/api/local-ai/*` and `/api/inference/telemetry` routes.

use super::errors::ApiError;
use crate::app_service as service;
use crate::inference::setup;
use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use std::sync::Arc;

type SharedState = Arc<AppState>;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/inference/telemetry", get(inference_telemetry))
        .route("/local-ai/setup-status", get(local_ai_setup_status))
        .route("/local-ai/cancel-download", post(cancel_model_download))
        .route("/local-ai/download-chat-model", post(setup_download_chat_model))
        .route("/local-ai/download-embed-model", post(setup_download_embed_model))
        .route("/local-ai/start-engine", post(setup_start_engine))
        .route("/local-ai/chat-model", delete(setup_remove_chat_model))
        .route("/local-ai/embed-model", delete(setup_remove_embed_model))
        .route("/local-ai/progress", get(local_ai_download_progress))
}

async fn inference_telemetry() -> Json<service::InferenceTelemetryView> {
    Json(service::inference_telemetry())
}

async fn local_ai_setup_status() -> Json<setup::LlamaSetupStatus> {
    Json(setup::local_ai_setup_status().await)
}

async fn cancel_model_download(State(state): State<SharedState>) -> Result<StatusCode, ApiError> {
    setup::cancel_model_download(&state)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Downloads are fire-and-forget from the HTTP caller's perspective: this
/// returns `202` immediately and the browser polls `/local-ai/progress` (and
/// `/local-ai/setup-status`) to track completion, since a multi-hundred-MB
/// download would otherwise hold the request open for minutes.
async fn setup_download_chat_model(State(state): State<SharedState>) -> StatusCode {
    tokio::spawn(async move {
        if let Err(error) = setup::setup_download_chat_model(state).await {
            tracing::error!(%error, "chat model download failed");
        }
    });
    StatusCode::ACCEPTED
}

async fn setup_download_embed_model(State(state): State<SharedState>) -> StatusCode {
    tokio::spawn(async move {
        if let Err(error) = setup::setup_download_embed_model(state).await {
            tracing::error!(%error, "embedding model download failed");
        }
    });
    StatusCode::ACCEPTED
}

async fn setup_start_engine(State(state): State<SharedState>) -> Result<StatusCode, ApiError> {
    setup::setup_start_engine(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn setup_remove_chat_model(State(state): State<SharedState>) -> Result<StatusCode, ApiError> {
    setup::setup_remove_chat_model(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn setup_remove_embed_model(State(state): State<SharedState>) -> Result<StatusCode, ApiError> {
    setup::setup_remove_embed_model(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn local_ai_download_progress(State(state): State<SharedState>) -> impl IntoResponse {
    match setup::download_progress(&state) {
        Some(progress) => Json(progress).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}
