//! `/api/events*` and `/api/data/*` routes.

use super::errors::{ApiError, ApiResult};
use crate::app_service as service;
use crate::persistence::sqlite::{RawEvent, SemanticEvent, SemanticEventView};
use crate::state::AppState;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::sync::Arc;

type SharedState = Arc<AppState>;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/events/recent-count", get(recent_event_count))
        .route("/events", get(list_events).post(record_event))
        .route("/events/semantic", get(list_semantic_events).post(record_semantic_event))
        .route("/events/raw-overview", get(list_raw_event_processing_overview))
        .route("/events/{id}/semantic", get(semantic_for_event))
        .route("/data/delete-all", post(delete_all_data))
        .route("/data/export", get(export_data))
        .route("/startup-diagnostics", get(startup_diagnostics))
}

async fn recent_event_count(State(state): State<SharedState>) -> ApiResult<i64> {
    Ok(Json(service::recent_event_count(&state).await?))
}

#[derive(Deserialize)]
struct ListQuery {
    limit: u32,
    query: Option<String>,
}

async fn list_events(State(state): State<SharedState>, Query(query): Query<ListQuery>) -> ApiResult<Vec<RawEvent>> {
    Ok(Json(service::list_events(&state, query.limit, query.query).await?))
}

async fn record_event(State(state): State<SharedState>, Json(event): Json<RawEvent>) -> Result<StatusCode, ApiError> {
    service::record_event(&state, event)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_semantic_events(
    State(state): State<SharedState>,
    Query(query): Query<ListQuery>,
) -> ApiResult<Vec<SemanticEventView>> {
    Ok(Json(service::list_semantic_events(&state, query.limit, query.query).await?))
}

async fn record_semantic_event(
    State(state): State<SharedState>,
    Json(event): Json<SemanticEvent>,
) -> Result<StatusCode, ApiError> {
    service::record_semantic_event(&state, event)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct OverviewQuery {
    limit: u32,
}

async fn list_raw_event_processing_overview(
    State(state): State<SharedState>,
    Query(query): Query<OverviewQuery>,
) -> ApiResult<Vec<service::RawEventProcessingOverview>> {
    Ok(Json(service::list_raw_event_processing_overview(&state, query.limit).await?))
}

async fn semantic_for_event(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> ApiResult<Option<SemanticEvent>> {
    Ok(Json(service::semantic_for_event(&state, id).await?))
}

async fn delete_all_data(State(state): State<SharedState>) -> Result<StatusCode, ApiError> {
    service::delete_all_data(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn export_data(State(state): State<SharedState>) -> ApiResult<String> {
    Ok(Json(service::export_data(&state).await?))
}

async fn startup_diagnostics(State(state): State<SharedState>) -> Json<Option<String>> {
    Json(service::startup_diagnostics(&state))
}
