//! HTTP surface for Chronicle's daemon binary.
//!
//! Chronicle runs as a single long-lived process (no installer, no native
//! app shell) that exposes its state over a local-only JSON API on
//! `127.0.0.1`. Each submodule here is a thin translation layer: handlers
//! extract the request, forward to a plain function in `app_service`/
//! `inference::setup` (the actual business logic, unaware of HTTP), and
//! serialize the result. See `lib.rs` for where this router is mounted
//! alongside the static frontend assets.

mod capture;
mod data_directory;
mod diagnostics;
mod errors;
mod events;
mod local_ai;
mod processing;
mod settings;

use crate::app_service;
use crate::state::AppState;
use axum::routing::get;
use axum::Router;
use std::sync::Arc;

type SharedState = Arc<AppState>;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/health", get(health_check))
        .merge(events::router())
        .merge(settings::router())
        .merge(capture::router())
        .merge(processing::router())
        .merge(data_directory::router())
        .merge(local_ai::router())
        .merge(diagnostics::router())
}

async fn health_check() -> &'static str {
    app_service::health_check()
}
