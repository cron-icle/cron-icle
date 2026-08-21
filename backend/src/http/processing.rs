//! `/api/processing/*` routes.

use super::errors::ApiResult;
use crate::app_service::{self as service, EventProcessingStatus, ProcessingMetricsView, ProcessingQueueLimits, ProcessingQueueStatus};
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use std::sync::Arc;

type SharedState = Arc<AppState>;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/processing/queue-status", get(processing_queue_status))
        .route("/processing/metrics", get(processing_metrics))
        .route("/processing/queue-limits", get(processing_queue_limits))
        .route("/processing/cancel-pending", post(cancel_pending_processing_tasks))
        .route("/processing/retry-failed", post(retry_failed_processing_tasks))
        .route("/processing/status/{event_id}", get(processing_status_for_event))
}

async fn processing_queue_status(State(state): State<SharedState>) -> ApiResult<ProcessingQueueStatus> {
    Ok(Json(service::processing_queue_status(&state).await?))
}

async fn processing_metrics(State(state): State<SharedState>) -> ApiResult<ProcessingMetricsView> {
    Ok(Json(service::processing_metrics(&state)?))
}

async fn processing_queue_limits() -> Json<ProcessingQueueLimits> {
    Json(service::processing_queue_limits())
}

async fn cancel_pending_processing_tasks(State(state): State<SharedState>) -> ApiResult<usize> {
    Ok(Json(service::cancel_pending_processing_tasks(&state).await?))
}

async fn retry_failed_processing_tasks(State(state): State<SharedState>) -> ApiResult<usize> {
    Ok(Json(service::retry_failed_processing_tasks(&state).await?))
}

async fn processing_status_for_event(
    State(state): State<SharedState>,
    Path(raw_event_id): Path<String>,
) -> ApiResult<Vec<EventProcessingStatus>> {
    Ok(Json(service::processing_status_for_event(&state, raw_event_id).await?))
}
