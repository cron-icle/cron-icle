//! `/api/diagnostics/*`, `/api/storage/usage`, `/api/model-provider/status`.

use super::errors::ApiResult;
use crate::app_service::{self as service, CaptureDiagnostics, ModelProviderStatus};
use crate::state::AppState;
use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use std::sync::Arc;

type SharedState = Arc<AppState>;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/storage/usage", get(storage_usage))
        .route("/model-provider/status", get(model_provider_status))
        .route("/diagnostics/capture", get(capture_diagnostics))
}

async fn storage_usage(State(state): State<SharedState>) -> ApiResult<std::collections::HashMap<String, i64>> {
    Ok(Json(service::storage_usage(&state).await?))
}

async fn model_provider_status() -> ApiResult<ModelProviderStatus> {
    Ok(Json(service::model_provider_status().await?))
}

async fn capture_diagnostics(State(state): State<SharedState>) -> ApiResult<CaptureDiagnostics> {
    Ok(Json(service::capture_diagnostics(&state).await?))
}
