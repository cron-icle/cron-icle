//! Application-level operations exposed over HTTP by the `http` module.
//!
//! Functions validate/clamp user-facing inputs and delegate work to the
//! database, capture lifecycle, and settings services. Long-running capture
//! work runs on background threads so request handlers stay responsive.
//! These functions are transport-agnostic (no `axum`/HTTP types) so they
//! stay callable from something other than an HTTP handler.

use crate::capture::activity::CaptureSettings;
use crate::inference::model_provider::{LlamaCppProvider, LocalModelStatus};
use crate::persistence::sqlite::{
    count_events_on, embedding_exists_on, processing_status_for_raw_event_on,
    queue_counts_on, recent_events_on, recent_semantic_events_on, semantic_for_raw_event_on,
    storage_counts_on, RawEvent, SemanticEvent, SemanticEventView,
};
use crate::processing::queue::{
    run_processing_worker_with_metrics, LocalModelQueueProcessor, MAX_PENDING_TASKS,
    MAX_RETRY_ATTEMPTS,
};
use crate::state::{AppState, DataDirectoryMoveProgress};
use serde::Serialize;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Debug, Clone, serde::Serialize)]
pub struct CaptureStatus {
    pub enabled: bool,
    pub foreground_provider_available: bool,
    pub active: bool,
    pub persisted_event_count: i64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProcessingQueueStatus {
    pub pending: i64,
    pub processing: i64,
    pub complete: i64,
    pub failed: i64,
    pub cancelled: i64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EventProcessingStatus {
    pub task_type: String,
    pub status: String,
    pub attempts: u32,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RawEventProcessingOverview {
    pub event: RawEvent,
    pub processing: Vec<EventProcessingStatus>,
    pub semantic_ready: bool,
    pub embedding_ready: bool,
}

pub fn health_check() -> &'static str {
    "ok"
}

pub fn capture_active_window_screenshot(window_handle: isize) -> Result<Vec<u8>, String> {
    #[cfg(windows)]
    {
        use crate::capture::screenshot::{
            ActiveWindowScreenshotProvider, PlatformActiveWindowScreenshotProvider,
        };
        return PlatformActiveWindowScreenshotProvider { window_handle }.capture_active_window();
    }
    #[cfg(not(windows))]
    {
        crate::capture::active_window::capture_window_png(window_handle)
    }
}

pub fn graphics_capture_session_available(window_handle: isize) -> Result<bool, String> {
    crate::capture::graphics_session::initialize(window_handle).map(|_session| true)
}

pub async fn recent_event_count(state: &AppState) -> Result<i64, String> {
    state.read_with(|connection| count_events_on(connection)).await
}

pub async fn list_events(
    state: &AppState,
    limit: u32,
    query: Option<String>,
) -> Result<Vec<RawEvent>, String> {
    let limit = limit.clamp(1, 500);
    state
        .read_with(move |connection| recent_events_on(connection, limit, query.as_deref()))
        .await
}

pub async fn list_semantic_events(
    state: &AppState,
    limit: u32,
    query: Option<String>,
) -> Result<Vec<SemanticEventView>, String> {
    let limit = limit.clamp(1, 500);
    state
        .read_with(move |connection| recent_semantic_events_on(connection, limit, query.as_deref()))
        .await
}

pub async fn list_raw_event_processing_overview(
    state: &AppState,
    limit: u32,
) -> Result<Vec<RawEventProcessingOverview>, String> {
    let limit = limit.clamp(1, 500);
    state
        .read_with(move |connection| {
            recent_events_on(connection, limit, None)?
                .into_iter()
                .map(|event| {
                    let semantic = semantic_for_raw_event_on(connection, &event.id)?;
                    let embedding_ready = match &semantic {
                        Some(value) => embedding_exists_on(connection, &value.id)?,
                        None => false,
                    };
                    let processing = processing_status_for_raw_event_on(connection, &event.id)?
                        .into_iter()
                        .map(
                            |(task_type, status, attempts, error)| EventProcessingStatus {
                                task_type,
                                status,
                                attempts,
                                error,
                            },
                        )
                        .collect();
                    Ok(RawEventProcessingOverview {
                        event,
                        processing,
                        semantic_ready: semantic.is_some(),
                        embedding_ready,
                    })
                })
                .collect()
        })
        .await
}

pub fn record_event(state: &AppState, event: RawEvent) -> Result<(), String> {
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .insert_event_and_enqueue(&event)
        .map_err(|error| error.to_string())
}

pub fn record_semantic_event(state: &AppState, event: SemanticEvent) -> Result<(), String> {
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .insert_semantic_event(&event)
        .map_err(|error| error.to_string())
}

pub async fn semantic_for_event(
    state: &AppState,
    raw_event_id: String,
) -> Result<Option<SemanticEvent>, String> {
    state
        .read_with(move |connection| semantic_for_raw_event_on(connection, &raw_event_id))
        .await
}

pub async fn delete_all_data(state: &AppState) -> Result<(), String> {
    let database = state.database.clone();
    tokio::task::spawn_blocking(move || {
        database
            .lock()
            .map_err(|_| "database lock poisoned".to_owned())?
            .delete_all()
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?
}

/// Reports whether database initialization degraded to an in-memory,
/// non-persistent database (see `AppState::initialize`), so the UI can show
/// a recoverable-error banner instead of the failure being silent.
pub fn startup_diagnostics(state: &AppState) -> Option<String> {
    state.startup_error.clone()
}

pub fn get_capture_settings(state: &AppState) -> Result<CaptureSettings, String> {
    Ok(state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?
        .clone())
}

pub fn update_capture_settings(
    state: &AppState,
    settings: CaptureSettings,
) -> Result<CaptureSettings, String> {
    let json = serde_json::to_string(&settings).map_err(|error| error.to_string())?;
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .save_setting("capture", &json)
        .map_err(|error| error.to_string())?;
    *state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())? = settings.clone();
    Ok(settings)
}

pub fn set_input_permission(
    state: &AppState,
    input: String,
    enabled: bool,
) -> Result<CaptureSettings, String> {
    let mut settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?;
    match input.as_str() {
        "keyboard" => settings.keyboard_enabled = enabled,
        "mouse" => settings.mouse_enabled = enabled,
        _ => return Err("input must be keyboard or mouse".to_owned()),
    }
    let json = serde_json::to_string(&*settings).map_err(|error| error.to_string())?;
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .save_setting("capture", &json)
        .map_err(|error| error.to_string())?;
    Ok(settings.clone())
}

pub fn set_screenshot_permission(state: &AppState, enabled: bool) -> Result<CaptureSettings, String> {
    let mut settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?;
    settings.screenshots_enabled = enabled;
    let json = serde_json::to_string(&*settings).map_err(|error| error.to_string())?;
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .save_setting("capture", &json)
        .map_err(|error| error.to_string())?;
    Ok(settings.clone())
}

pub fn set_keyboard_text_allowlist(
    state: &AppState,
    applications: Vec<String>,
) -> Result<CaptureSettings, String> {
    let mut settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?;
    settings.keyboard_text_allowlist = applications
        .into_iter()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .collect();
    settings.keyboard_mode = if settings.keyboard_text_allowlist.is_empty() {
        crate::capture::activity::KeyboardMode::MetadataOnly
    } else {
        crate::capture::activity::KeyboardMode::AllowlistedText
    };
    let json = serde_json::to_string(&*settings).map_err(|error| error.to_string())?;
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .save_setting("capture", &json)
        .map_err(|error| error.to_string())?;
    Ok(settings.clone())
}

pub fn set_excluded_applications(
    state: &AppState,
    applications: Vec<String>,
) -> Result<CaptureSettings, String> {
    let mut settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?;
    let mut normalized = Vec::new();
    for application in applications {
        let value = application.trim().to_ascii_lowercase();
        if !value.is_empty() && !normalized.contains(&value) {
            normalized.push(value);
        }
    }
    settings.excluded_applications = normalized;
    let json = serde_json::to_string(&*settings).map_err(|error| error.to_string())?;
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .save_setting("capture", &json)
        .map_err(|error| error.to_string())?;
    Ok(settings.clone())
}

pub fn set_excluded_paths(state: &AppState, paths: Vec<String>) -> Result<CaptureSettings, String> {
    let mut settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?;
    settings.excluded_paths = paths
        .into_iter()
        .map(|path| path.trim().to_owned())
        .filter(|path| !path.is_empty())
        .collect();
    let json = serde_json::to_string(&*settings).map_err(|error| error.to_string())?;
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .save_setting("capture", &json)
        .map_err(|error| error.to_string())?;
    Ok(settings.clone())
}

pub fn set_watched_folders(state: &AppState, folders: Vec<String>) -> Result<CaptureSettings, String> {
    let mut settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?;
    let mut normalized = Vec::new();
    for folder in folders {
        let value = folder.trim().to_owned();
        if !value.is_empty()
            && std::path::Path::new(&value).is_dir()
            && !normalized.contains(&value)
        {
            normalized.push(value);
        }
    }
    settings.watched_folders = normalized;
    let json = serde_json::to_string(&*settings).map_err(|error| error.to_string())?;
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .save_setting("capture", &json)
        .map_err(|error| error.to_string())?;
    Ok(settings.clone())
}

pub async fn export_data(state: &AppState) -> Result<String, String> {
    let database = state.database.clone();
    tokio::task::spawn_blocking(move || {
        database
            .lock()
            .map_err(|_| "database lock poisoned".to_owned())?
            .export_json()
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?
}

pub fn start_capture_state(state: &AppState) -> Result<(), String> {
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .enqueue_unprocessed_events(500)
        .map_err(|error| error.to_string())?;
    let mut stop_slot = state
        .capture_stop
        .lock()
        .map_err(|_| "capture lock poisoned".to_owned())?;
    if stop_slot.is_some() {
        return Ok(());
    }
    let stop = Arc::new(AtomicBool::new(false));
    let keyboard_enabled = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?
        .keyboard_enabled;
    let mouse_enabled = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?
        .mouse_enabled;
    let thread = crate::capture::activity::start_foreground_loop(
        state.database.clone(),
        stop.clone(),
        state.settings.clone(),
        state.screenshot_cache.clone(),
    );
    *stop_slot = Some(stop.clone());
    let mut threads = state
        .capture_threads
        .lock()
        .map_err(|_| "capture thread lock poisoned".to_owned())?;
    threads.push(thread);
    if let Ok(mut metrics) = state.processing_metrics.lock() {
        metrics.reset();
    }
    threads.push(run_processing_worker_with_metrics(
        state.database.clone(),
        stop.clone(),
        Arc::new(LocalModelQueueProcessor {
            database: state.database.clone(),
            screenshot_cache: state.screenshot_cache.clone(),
        }),
        state.processing_metrics.clone(),
    ));
    threads.push(crate::filesystem_watch::start_filesystem_loop(
        state.database.clone(),
        stop.clone(),
        state.settings.clone(),
    ));
    // Mouse/keyboard hook callbacks must never block on the database lock
    // (that would stall the OS message pump and visibly lag input for the
    // whole system), so they send normalized events down a channel to a
    // single batching writer thread instead of writing directly.
    let (writer_sender, writer_receiver) = std::sync::mpsc::channel();
    threads.push(crate::persistence::writer::start_capture_writer(
        state.database.clone(),
        state.settings.clone(),
        writer_receiver,
    ));
    #[cfg(windows)]
    if mouse_enabled {
        threads.push(crate::capture::input::start_mouse_hook(
            writer_sender.clone(),
            stop.clone(),
        ));
    }
    #[cfg(windows)]
    if keyboard_enabled {
        threads.push(crate::capture::input::start_keyboard_hook(
            writer_sender.clone(),
            stop.clone(),
        ));
    }
    let mut settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?;
    settings.enabled = true;
    let settings_json = serde_json::to_string(&*settings).map_err(|error| error.to_string())?;
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .save_setting("capture", &settings_json)
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub fn start_capture(state: &AppState) -> Result<(), String> {
    start_capture_state(state)
}

pub async fn capture_status(state: &AppState) -> Result<CaptureStatus, String> {
    let enabled = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?
        .enabled;
    let active = state
        .capture_stop
        .lock()
        .map_err(|_| "capture lock poisoned".to_owned())?
        .is_some();
    let persisted_event_count = state.read_with(|connection| count_events_on(connection)).await?;
    Ok(CaptureStatus {
        enabled,
        active,
        foreground_provider_available: cfg!(windows),
        persisted_event_count,
    })
}

fn to_queue_status(counts: std::collections::HashMap<String, i64>) -> ProcessingQueueStatus {
    ProcessingQueueStatus {
        pending: *counts.get("pending").unwrap_or(&0),
        processing: *counts.get("processing").unwrap_or(&0),
        complete: *counts.get("complete").unwrap_or(&0),
        failed: *counts.get("failed").unwrap_or(&0),
        cancelled: *counts.get("cancelled").unwrap_or(&0),
    }
}

pub async fn processing_queue_status(state: &AppState) -> Result<ProcessingQueueStatus, String> {
    let counts = state.read_with(|connection| queue_counts_on(connection)).await?;
    Ok(to_queue_status(counts))
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessingMetricsView {
    pub completed: u64,
    pub failed: u64,
    pub panicked: u64,
    pub average_latency_ms: Option<f64>,
    pub last_model_name: Option<String>,
    pub last_model_version: Option<String>,
}

/// Live throughput/latency/failure counters for the AI processing worker —
/// the "is the pipeline actually keeping up" view that queue counts alone
/// can't answer (a growing `pending` count could mean the worker is idle,
/// crashing, or just slow; this distinguishes them).
pub fn processing_metrics(state: &AppState) -> Result<ProcessingMetricsView, String> {
    let metrics = state
        .processing_metrics
        .lock()
        .map_err(|_| "processing metrics lock poisoned".to_owned())?
        .snapshot();
    Ok(ProcessingMetricsView {
        completed: metrics.completed,
        failed: metrics.failed,
        panicked: metrics.panicked,
        average_latency_ms: metrics.average_latency_ms(),
        last_model_name: metrics.last_model_name,
        last_model_version: metrics.last_model_version,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareProfileView {
    pub logical_cores: usize,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub gpu_available: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferenceTelemetryView {
    pub hardware: HardwareProfileView,
    /// "unloaded" | "ready" | "idle" — the generation (chat/vision) model's
    /// current residency state. The embedding model has no separate state to
    /// report here since it never idle-unloads by design — it's either not
    /// yet loaded or resident, which `processing_metrics`' `last_model_name`
    /// already implies once anything has processed.
    pub generation_engine_state: &'static str,
    pub current_batch_size: usize,
}

/// Snapshot of the hardware/memory/model-lifecycle state behind the AI
/// pipeline's scheduling decisions — "why is it batching this many items,
/// why did the model just reload" made visible instead of only inferable
/// from timing. Complements `processing_metrics` (throughput/latency) and
/// `processing_queue_status` (backlog) with the resource picture behind
/// both.
pub fn inference_telemetry() -> InferenceTelemetryView {
    let profile = crate::hardware_profiler::HardwareProfile::detect();
    let current_batch_size = crate::memory_planner::adaptive_batch_size(
        crate::processing::queue::MAX_MODEL_BATCH_SIZE,
        &profile,
    );
    InferenceTelemetryView {
        hardware: HardwareProfileView {
            logical_cores: profile.logical_cores,
            total_ram_mb: profile.total_ram_bytes / (1024 * 1024),
            available_ram_mb: profile.available_ram_bytes / (1024 * 1024),
            gpu_available: profile.gpu.is_some(),
        },
        generation_engine_state: match crate::inference::native::generation_engine_state() {
            crate::inference::native::ModelState::Unloaded => "unloaded",
            crate::inference::native::ModelState::Ready => "ready",
            crate::inference::native::ModelState::Idle => "idle",
        },
        current_batch_size,
    }
}

pub async fn storage_usage(state: &AppState) -> Result<std::collections::HashMap<String, i64>, String> {
    state.read_with(|connection| storage_counts_on(connection)).await
}

#[derive(Debug, Serialize)]
pub struct ModelProviderStatus {
    pub semantic_provider: String,
    pub embedding_provider: String,
    pub semantic_available: bool,
    pub embedding_available: bool,
    pub local_models: LocalModelStatus,
}

fn model_provider_status_blocking() -> ModelProviderStatus {
    let provider = LlamaCppProvider::default();
    let local_models = provider.status();
    ModelProviderStatus {
        semantic_provider: format!("llama.cpp/Gemma ({})", local_models.chat_model),
        embedding_provider: format!("llama.cpp/EmbeddingGemma ({})", local_models.embedding_model),
        semantic_available: local_models.chat_available,
        embedding_available: local_models.embedding_available,
        local_models,
    }
}

/// Queries local model availability, which involves a blocking HTTP call to
/// the local llama.cpp engine. Runs off the async executor via
/// `spawn_blocking`.
pub async fn model_provider_status() -> Result<ModelProviderStatus, String> {
    tokio::task::spawn_blocking(model_provider_status_blocking)
        .await
        .map_err(|error| error.to_string())
}

#[derive(Debug, Serialize)]
pub struct ProcessingQueueLimits {
    pub max_retry_attempts: u32,
    pub max_pending_tasks: u32,
}

pub fn processing_queue_limits() -> ProcessingQueueLimits {
    ProcessingQueueLimits {
        max_retry_attempts: MAX_RETRY_ATTEMPTS,
        max_pending_tasks: MAX_PENDING_TASKS,
    }
}

#[derive(Debug, Serialize)]
pub struct CaptureDiagnostics {
    pub settings: CaptureSettings,
    pub storage: std::collections::HashMap<String, i64>,
    pub queue: ProcessingQueueStatus,
    pub providers: ModelProviderStatus,
}

pub async fn capture_diagnostics(state: &AppState) -> Result<CaptureDiagnostics, String> {
    let settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?
        .clone();
    let storage = state.read_with(|connection| storage_counts_on(connection)).await?;
    let counts = state.read_with(|connection| queue_counts_on(connection)).await?;
    let providers = tokio::task::spawn_blocking(model_provider_status_blocking)
        .await
        .map_err(|error| error.to_string())?;
    Ok(CaptureDiagnostics {
        settings,
        storage,
        queue: to_queue_status(counts),
        providers,
    })
}

pub async fn cancel_pending_processing_tasks(state: &AppState) -> Result<usize, String> {
    let database = state.database.clone();
    tokio::task::spawn_blocking(move || {
        database
            .lock()
            .map_err(|_| "database lock poisoned".to_owned())?
            .cancel_pending_tasks()
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?
}

pub async fn retry_failed_processing_tasks(state: &AppState) -> Result<usize, String> {
    let database = state.database.clone();
    tokio::task::spawn_blocking(move || {
        database
            .lock()
            .map_err(|_| "database lock poisoned".to_owned())?
            .retry_failed_tasks()
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?
}

pub async fn processing_status_for_event(
    state: &AppState,
    raw_event_id: String,
) -> Result<Vec<EventProcessingStatus>, String> {
    let rows = state
        .read_with(move |connection| processing_status_for_raw_event_on(connection, &raw_event_id))
        .await?;
    Ok(rows
        .into_iter()
        .map(
            |(task_type, status, attempts, error)| EventProcessingStatus {
                task_type,
                status,
                attempts,
                error,
            },
        )
        .collect())
}

pub fn stop_capture_state(state: &AppState) {
    if let Ok(mut stop_slot) = state.capture_stop.lock() {
        if let Some(stop) = stop_slot.take() {
            stop.store(true, Ordering::Relaxed);
        }
    }
    if let Ok(mut thread_slot) = state.capture_threads.lock() {
        for thread in thread_slot.drain(..) {
            let _ = thread.join();
        }
    }
}

pub fn shutdown_llama_engine(state: &AppState) {
    for slot in [&state.llama_chat_process, &state.llama_embed_process] {
        if let Ok(mut process_slot) = slot.lock() {
            if let Some(mut process) = process_slot.take() {
                let _ = process.kill();
                let _ = process.wait();
            }
        }
    }
}

pub fn stop_capture(state: &AppState) -> Result<(), String> {
    stop_capture_state(state);
    let mut settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?;
    settings.enabled = false;
    let settings_json = serde_json::to_string(&*settings).map_err(|error| error.to_string())?;
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .save_setting("capture", &settings_json)
        .map_err(|error| error.to_string())?;
    Ok(())
}

/// The directory Chronicle currently stores its data (database, downloaded
/// models) under, or `None` if the user hasn't chosen one yet.
pub fn get_data_directory() -> Option<String> {
    crate::data_directory::current().map(|dir| dir.display().to_string())
}

/// Current progress of an in-flight data-directory move, if any. Polled by
/// the UI instead of pushed, since there is no event channel — the app runs
/// as a plain HTTP server.
pub fn data_directory_move_progress(state: &AppState) -> Option<DataDirectoryMoveProgress> {
    state.data_dir_move_progress.lock().ok().and_then(|guard| guard.clone())
}

/// Picks (if none is set yet) or moves (if one already is) Chronicle's data
/// directory, then relaunches the process so it starts fresh against it.
///
/// Everything that could hold a file open under the current data directory
/// is stopped first — capture threads, both llama.cpp servers, and the
/// active sqlite connection is checkpointed (WAL merged into the main file)
/// so the copy underneath `data_directory::relocate_or_set` sees a
/// consistent, unlocked-as-possible set of files. There is no partial/hot-swap
/// path: like first-run resolution, this refuses to guess, and the only way
/// the new location actually takes effect is a full relaunch that
/// re-resolves `data_directory::current()` from the updated pointer file.
///
/// Runs the move synchronously and returns once it's done; the caller (the
/// HTTP handler) is responsible for relaunching the process and exiting
/// after the HTTP response has had a chance to flush.
pub async fn change_data_directory(state: Arc<AppState>) -> Result<(), String> {
    stop_capture_state(&state);
    shutdown_llama_engine(&state);
    if let Ok(database) = state.database.lock() {
        let _ = database.checkpoint_wal();
    }
    *state.data_dir_move_progress.lock().map_err(|_| "progress lock poisoned".to_owned())? = None;
    let progress_state = state.clone();
    tokio::task::spawn_blocking(move || {
        let chosen = rfd::FileDialog::new()
            .set_title("Choose a new folder for Chronicle to store its data and downloaded models")
            .pick_folder()
            .ok_or_else(|| "no folder was chosen".to_string())?;
        let mut last_emit = std::time::Instant::now() - std::time::Duration::from_secs(1);
        crate::data_directory::relocate_or_set(&chosen, move |copied, total| {
            if last_emit.elapsed() < std::time::Duration::from_millis(200) && copied < total {
                return;
            }
            last_emit = std::time::Instant::now();
            let percent = if total > 0 {
                (copied as f64 / total as f64 * 100.0) as f32
            } else {
                100.0
            };
            if let Ok(mut slot) = progress_state.data_dir_move_progress.lock() {
                *slot = Some(DataDirectoryMoveProgress {
                    copied_bytes: copied,
                    total_bytes: total,
                    percent,
                });
            }
        })
    })
    .await
    .map_err(|error| error.to_string())??;
    if let Ok(mut slot) = state.data_dir_move_progress.lock() {
        *slot = None;
    }
    Ok(())
}

/// Relaunches the current executable as a fresh, detached process. Callers
/// should exit(0) shortly after so the HTTP response announcing the restart
/// has time to flush to the browser first.
pub fn relaunch_self() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|error| error.to_string())?;
    std::process::Command::new(exe)
        .spawn()
        .map_err(|error| error.to_string())?;
    Ok(())
}
