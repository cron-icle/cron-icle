//! `/api/data-directory*` routes.

use super::errors::ApiResult;
use crate::app_service as service;
use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use std::sync::Arc;

type SharedState = Arc<AppState>;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/data-directory", get(get_data_directory))
        .route("/data-directory/change", post(change_data_directory))
        .route("/data-directory/move-progress", get(data_directory_move_progress))
}

async fn get_data_directory() -> Json<Option<String>> {
    Json(service::get_data_directory())
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
    service::change_data_directory(state.clone()).await?;
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        if let Err(error) = service::relaunch_self() {
            tracing::error!(%error, "failed to relaunch after data directory change");
        }
        std::process::exit(0);
    });
    Ok(Json(Restarting { restarting: true }))
}

async fn data_directory_move_progress(State(state): State<SharedState>) -> impl IntoResponse {
    match service::data_directory_move_progress(&state) {
        Some(progress) => Json(progress).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}
