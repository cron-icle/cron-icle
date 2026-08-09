//! Tauri IPC commands exposed to the desktop UI.
//!
//! Commands validate/clamp user-facing inputs and delegate work to the
//! database, capture lifecycle, and settings services. Long-running capture
//! work is launched in a background thread so invoke handlers stay responsive.

use crate::activity_capture::CaptureSettings;
use crate::asynchronous_processing_queue::{
    run_processing_worker_with_metrics, LocalModelQueueProcessor, ProcessingMetrics, MAX_PENDING_TASKS,
    MAX_RETRY_ATTEMPTS,
};
use crate::local_model_provider::{LlamaCppProvider, LocalModelStatus};
use crate::local_sqlite_event_database::{
    self, count_events_on, embedding_exists_on, processing_status_for_raw_event_on,
    queue_counts_on, recent_events_on, recent_semantic_events_on, semantic_for_raw_event_on,
    storage_counts_on, Database, ReaderPool, RawEvent, SemanticEvent, SemanticEventView,
};
use serde::Serialize;
use std::process::Child;
use std::sync::Mutex;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::JoinHandle;
use tauri::{Emitter, State};

pub struct AppState {
    pub database: Arc<Mutex<Database>>,
    /// Read-only connection pool for UI query commands (list/search/count/
    /// diagnostics), kept separate from the single writer connection above so
    /// reads never contend with capture writes or each other.
    pub reader_pool: Option<ReaderPool>,
    pub settings: Arc<Mutex<CaptureSettings>>,
    pub capture_stop: Mutex<Option<Arc<AtomicBool>>>,
    pub capture_threads: Mutex<Vec<JoinHandle<()>>>,
    /// The chat/vision `llama-server` process, if Chronicle started it.
    pub llama_chat_process: Mutex<Option<Child>>,
    /// The embedding `llama-server` process, if Chronicle started it.
    pub llama_embed_process: Mutex<Option<Child>>,
    /// Set when database initialization failed and a degraded in-memory
    /// database is being used instead. Surfaced to the UI so a failed disk
    /// database is a visible, recoverable error state rather than a crash.
    pub startup_error: Option<String>,
    /// Bounded, memory-only holding area for screenshots captured at focus
    /// change so the AI queue can analyze the frame the user actually saw.
    pub screenshot_cache: Arc<Mutex<crate::capture_writer::ScreenshotCache>>,
    /// Set to request the in-flight model download (if any) stop at its next
    /// checked point. Reset to `false` at the start of each new download.
    pub download_cancel: Arc<AtomicBool>,
    /// Throughput/latency/failure counters for the AI processing worker,
    /// updated live as it runs. Surfaced to the UI via `processing_metrics`
    /// so pipeline health (is it keeping up, is it failing, how slow is
    /// each batch) is observable instead of only inferable from queue
    /// counts after the fact.
    pub processing_metrics: Arc<Mutex<ProcessingMetrics>>,
}

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

impl AppState {
    /// Builds application state. A database open failure is a recoverable
    /// condition, not a process crash: this falls back to a transient
    /// in-memory database (capture keeps working, just without persistence
    /// across restarts) and records the failure in `startup_error` so the UI
    /// can surface it instead of the app disappearing silently.
    pub fn initialize() -> Self {
        // No data directory chosen yet is not a failure: the user picks one
        // from Settings when they set up local AI, and until then Chronicle
        // simply runs on a transient in-memory database rather than
        // blocking startup on a folder-choose dialog.
        let data_dir_configured = crate::data_directory::current().is_some();
        let (database, startup_error) = if !data_dir_configured {
            (
                Database::open_in_memory_degraded()
                    .expect("in-memory sqlite connection is expected to always succeed"),
                None,
            )
        } else {
            match Database::open() {
                Ok(database) => (database, None),
                Err(error) => {
                    tracing::error!(%error, "database initialization failed; falling back to an in-memory database");
                    let fallback = Database::open_in_memory_degraded().unwrap_or_else(|fallback_error| {
                        // Even the in-memory fallback failed. This should be
                        // effectively impossible (no I/O involved), but rather
                        // than panic we degrade further: continue with no
                        // persistence rather than crash the desktop shell.
                        tracing::error!(%fallback_error, "in-memory fallback database also failed to initialize");
                        Database::open_in_memory_degraded()
                            .expect("in-memory sqlite connection is expected to always succeed")
                    });
                    (
                        fallback,
                        Some(format!(
                            "Chronicle could not open its local database and is running in a temporary, non-persistent mode: {error}"
                        )),
                    )
                }
            }
        };
        if let Err(error) = database.seed_ready_event() {
            tracing::warn!(%error, "failed to seed initial ready event");
        }
        let settings = database
            .load_setting("capture")
            .ok()
            .flatten()
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default();
        // Nothing to start yet if no data directory is configured: the
        // engine binary is bundled, but the model files it would load live
        // under the (not yet chosen) data directory.
        let engine = LlamaCppProvider::default();
        let llama_chat_process = if data_dir_configured {
            engine.start_chat_server_if_needed().unwrap_or_else(|error| {
                tracing::warn!(%error, "chat/vision engine was not started; local AI will retry through the queue");
                None
            })
        } else {
            None
        };
        let llama_embed_process = if data_dir_configured {
            engine.start_embed_server_if_needed().unwrap_or_else(|error| {
                tracing::warn!(%error, "embedding engine was not started; local AI will retry through the queue");
                None
            })
        } else {
            None
        };
        // Only pool connections to the on-disk file: when we fell back to an
        // in-memory writer, a pooled reader would open an unrelated (or
        // stale) chronicle.db file rather than the live in-memory database.
        let reader_pool = if data_dir_configured && startup_error.is_none() {
            match local_sqlite_event_database::open_reader_pool() {
                Ok(pool) => Some(pool),
                Err(error) => {
                    tracing::warn!(%error, "failed to open read-only connection pool; read commands will fall back to the writer connection");
                    None
                }
            }
        } else {
            None
        };
        Self {
            database: Arc::new(Mutex::new(database)),
            reader_pool,
            settings: Arc::new(Mutex::new(settings)),
            capture_stop: Mutex::new(None),
            capture_threads: Mutex::new(Vec::new()),
            llama_chat_process: Mutex::new(llama_chat_process),
            llama_embed_process: Mutex::new(llama_embed_process),
            startup_error,
            screenshot_cache: Arc::new(Mutex::new(crate::capture_writer::ScreenshotCache::new(32))),
            download_cancel: Arc::new(AtomicBool::new(false)),
            processing_metrics: Arc::new(Mutex::new(ProcessingMetrics::default())),
        }
    }

    /// Runs a read-only query off the UI thread. Prefers a pooled read-only
    /// connection (see `ReaderPool`) so UI queries never contend with the
    /// single writer mutex used by capture and processing; falls back to the
    /// writer connection when no pool is available (e.g. degraded in-memory
    /// startup). Always runs inside `spawn_blocking` so the async command
    /// handler, and therefore the UI thread, is never blocked on rusqlite I/O.
    async fn read_with<T, F>(&self, query: F) -> Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce(&rusqlite::Connection) -> rusqlite::Result<T> + Send + 'static,
    {
        if let Some(pool) = self.reader_pool.clone() {
            tauri::async_runtime::spawn_blocking(move || {
                let connection = pool.get().map_err(|error| error.to_string())?;
                query(&connection).map_err(|error| error.to_string())
            })
            .await
            .map_err(|error| error.to_string())?
        } else {
            let database = self.database.clone();
            tauri::async_runtime::spawn_blocking(move || {
                let database = database
                    .lock()
                    .map_err(|_| "database lock poisoned".to_owned())?;
                query(database.connection()).map_err(|error| error.to_string())
            })
            .await
            .map_err(|error| error.to_string())?
        }
    }
}

#[tauri::command]
pub fn health_check() -> &'static str {
    "ok"
}

#[tauri::command]
pub fn capture_active_window_screenshot(window_handle: isize) -> Result<Vec<u8>, String> {
    #[cfg(windows)]
    {
        use crate::transient_screenshot_capture::{
            ActiveWindowScreenshotProvider, PlatformActiveWindowScreenshotProvider,
        };
        return PlatformActiveWindowScreenshotProvider { window_handle }.capture_active_window();
    }
    #[cfg(not(windows))]
    {
        crate::windows_active_window_screenshot::capture_window_png(window_handle)
    }
}

#[tauri::command]
pub fn graphics_capture_session_available(window_handle: isize) -> Result<bool, String> {
    crate::windows_graphics_capture_session::initialize(window_handle).map(|capture| {
        let _ = (&capture.frame_pool, &capture.session);
        true
    })
}

#[tauri::command]
pub async fn recent_event_count(state: State<'_, AppState>) -> Result<i64, String> {
    state.read_with(|connection| count_events_on(connection)).await
}

#[tauri::command]
pub async fn list_events(
    state: State<'_, AppState>,
    limit: u32,
    query: Option<String>,
) -> Result<Vec<RawEvent>, String> {
    let limit = limit.clamp(1, 500);
    state
        .read_with(move |connection| recent_events_on(connection, limit, query.as_deref()))
        .await
}

#[tauri::command]
pub async fn list_semantic_events(
    state: State<'_, AppState>,
    limit: u32,
    query: Option<String>,
) -> Result<Vec<SemanticEventView>, String> {
    let limit = limit.clamp(1, 500);
    state
        .read_with(move |connection| recent_semantic_events_on(connection, limit, query.as_deref()))
        .await
}

#[tauri::command]
pub async fn list_raw_event_processing_overview(
    state: State<'_, AppState>,
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

#[tauri::command]
pub fn record_event(state: State<'_, AppState>, event: RawEvent) -> Result<(), String> {
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .insert_event_and_enqueue(&event)
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn record_semantic_event(
    state: State<'_, AppState>,
    event: SemanticEvent,
) -> Result<(), String> {
    state
        .database
        .lock()
        .map_err(|_| "database lock poisoned".to_owned())?
        .insert_semantic_event(&event)
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub async fn semantic_for_event(
    state: State<'_, AppState>,
    raw_event_id: String,
) -> Result<Option<SemanticEvent>, String> {
    state
        .read_with(move |connection| semantic_for_raw_event_on(connection, &raw_event_id))
        .await
}

#[tauri::command]
pub async fn delete_all_data(state: State<'_, AppState>) -> Result<(), String> {
    let database = state.database.clone();
    tauri::async_runtime::spawn_blocking(move || {
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
#[tauri::command]
pub fn startup_diagnostics(state: State<'_, AppState>) -> Option<String> {
    state.startup_error.clone()
}

#[tauri::command]
pub fn get_capture_settings(state: State<'_, AppState>) -> Result<CaptureSettings, String> {
    Ok(state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?
        .clone())
}

#[tauri::command]
pub fn update_capture_settings(
    state: State<'_, AppState>,
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

#[tauri::command]
pub fn set_input_permission(
    state: State<'_, AppState>,
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

#[tauri::command]
pub fn set_screenshot_permission(
    state: State<'_, AppState>,
    enabled: bool,
) -> Result<CaptureSettings, String> {
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

#[tauri::command]
pub fn set_keyboard_text_allowlist(
    state: State<'_, AppState>,
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
        crate::activity_capture::KeyboardMode::MetadataOnly
    } else {
        crate::activity_capture::KeyboardMode::AllowlistedText
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

#[tauri::command]
pub fn set_excluded_applications(
    state: State<'_, AppState>,
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

#[tauri::command]
pub fn set_excluded_paths(
    state: State<'_, AppState>,
    paths: Vec<String>,
) -> Result<CaptureSettings, String> {
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

#[tauri::command]
pub fn set_watched_folders(
    state: State<'_, AppState>,
    folders: Vec<String>,
) -> Result<CaptureSettings, String> {
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

#[tauri::command]
pub async fn export_data(state: State<'_, AppState>) -> Result<String, String> {
    let database = state.database.clone();
    tauri::async_runtime::spawn_blocking(move || {
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
    let thread = crate::activity_capture::start_foreground_loop(
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
    threads.push(crate::filesystem_activity_capture::start_filesystem_loop(
        state.database.clone(),
        stop.clone(),
        state.settings.clone(),
    ));
    // Mouse/keyboard hook callbacks must never block on the database lock
    // (that would stall the OS message pump and visibly lag input for the
    // whole system), so they send normalized events down a channel to a
    // single batching writer thread instead of writing directly.
    let (writer_sender, writer_receiver) = std::sync::mpsc::channel();
    threads.push(crate::capture_writer::start_capture_writer(
        state.database.clone(),
        state.settings.clone(),
        writer_receiver,
    ));
    #[cfg(windows)]
    if mouse_enabled {
        threads.push(crate::input_capture::start_mouse_hook(
            writer_sender.clone(),
            stop.clone(),
        ));
    }
    #[cfg(windows)]
    if keyboard_enabled {
        threads.push(crate::input_capture::start_keyboard_hook(
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

#[tauri::command]
pub fn start_capture(state: State<'_, AppState>) -> Result<(), String> {
    start_capture_state(&state)
}

#[tauri::command]
pub async fn capture_status(state: State<'_, AppState>) -> Result<CaptureStatus, String> {
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

#[tauri::command]
pub async fn processing_queue_status(
    state: State<'_, AppState>,
) -> Result<ProcessingQueueStatus, String> {
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
#[tauri::command]
pub fn processing_metrics(state: State<'_, AppState>) -> Result<ProcessingMetricsView, String> {
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
    /// current residency state (see `native_inference::ModelState`). The
    /// embedding model has no separate state to report here since it never
    /// idle-unloads by design (see `native_inference` module docs) — it's
    /// either not yet loaded or resident, which `processing_metrics`'
    /// `last_model_name` already implies once anything has processed.
    pub generation_engine_state: &'static str,
    pub current_batch_size: usize,
}

/// Snapshot of the hardware/memory/model-lifecycle state behind the AI
/// pipeline's scheduling decisions — "why is it batching this many items,
/// why did the model just reload" made visible instead of only inferable
/// from timing. Complements `processing_metrics` (throughput/latency) and
/// `processing_queue_status` (backlog) with the resource picture behind
/// both.
#[tauri::command]
pub fn inference_telemetry() -> InferenceTelemetryView {
    let profile = crate::hardware_profiler::HardwareProfile::detect();
    let current_batch_size = crate::memory_planner::adaptive_batch_size(
        crate::asynchronous_processing_queue::MAX_MODEL_BATCH_SIZE,
        &profile,
    );
    InferenceTelemetryView {
        hardware: HardwareProfileView {
            logical_cores: profile.logical_cores,
            total_ram_mb: profile.total_ram_bytes / (1024 * 1024),
            available_ram_mb: profile.available_ram_bytes / (1024 * 1024),
            gpu_available: profile.gpu.is_some(),
        },
        generation_engine_state: match crate::native_inference::generation_engine_state() {
            crate::native_inference::ModelState::Unloaded => "unloaded",
            crate::native_inference::ModelState::Ready => "ready",
            crate::native_inference::ModelState::Idle => "idle",
        },
        current_batch_size,
    }
}

#[tauri::command]
pub async fn storage_usage(
    state: State<'_, AppState>,
) -> Result<std::collections::HashMap<String, i64>, String> {
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
/// the local llama.cpp engine. Runs off the UI thread via `spawn_blocking`.
#[tauri::command]
pub async fn model_provider_status() -> Result<ModelProviderStatus, String> {
    tauri::async_runtime::spawn_blocking(model_provider_status_blocking)
        .await
        .map_err(|error| error.to_string())
}

#[derive(Debug, Serialize)]
pub struct ProcessingQueueLimits {
    pub max_retry_attempts: u32,
    pub max_pending_tasks: u32,
}

#[tauri::command]
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

#[tauri::command]
pub async fn capture_diagnostics(state: State<'_, AppState>) -> Result<CaptureDiagnostics, String> {
    let settings = state
        .settings
        .lock()
        .map_err(|_| "settings lock poisoned".to_owned())?
        .clone();
    let storage = state.read_with(|connection| storage_counts_on(connection)).await?;
    let counts = state.read_with(|connection| queue_counts_on(connection)).await?;
    let providers =
        tauri::async_runtime::spawn_blocking(model_provider_status_blocking)
            .await
            .map_err(|error| error.to_string())?;
    Ok(CaptureDiagnostics {
        settings,
        storage,
        queue: to_queue_status(counts),
        providers,
    })
}

#[tauri::command]
pub async fn cancel_pending_processing_tasks(state: State<'_, AppState>) -> Result<usize, String> {
    let database = state.database.clone();
    tauri::async_runtime::spawn_blocking(move || {
        database
            .lock()
            .map_err(|_| "database lock poisoned".to_owned())?
            .cancel_pending_tasks()
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn retry_failed_processing_tasks(state: State<'_, AppState>) -> Result<usize, String> {
    let database = state.database.clone();
    tauri::async_runtime::spawn_blocking(move || {
        database
            .lock()
            .map_err(|_| "database lock poisoned".to_owned())?
            .retry_failed_tasks()
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn processing_status_for_event(
    state: State<'_, AppState>,
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

#[tauri::command]
pub fn stop_capture(state: State<'_, AppState>) -> Result<(), String> {
    stop_capture_state(&state);
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
#[tauri::command]
pub fn get_data_directory() -> Option<String> {
    crate::data_directory::current().map(|dir| dir.display().to_string())
}

#[derive(Debug, Clone, Serialize)]
pub struct DataDirectoryMoveProgress {
    pub copied_bytes: u64,
    pub total_bytes: u64,
    pub percent: f32,
}

/// Picks (if none is set yet) or moves (if one already is) Chronicle's data
/// directory, then relaunches the app so it opens fresh against it.
///
/// Everything that could hold a file open under the current data directory
/// is stopped first — capture threads, both llama.cpp servers, and the
/// active sqlite connection is checkpointed (WAL merged into the main file)
/// so the copy underneath `data_directory::relocate_or_set` sees a
/// consistent, unlocked-as-possible set of files. There is no partial/hot-swap
/// path: like first-run resolution, this refuses to guess, and the only way
/// the new location actually takes effect is a full relaunch that
/// re-resolves `data_directory::current()` from the updated pointer file.
#[tauri::command]
pub async fn change_data_directory(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    stop_capture_state(&state);
    shutdown_llama_engine(&state);
    if let Ok(database) = state.database.lock() {
        let _ = database.checkpoint_wal();
    }
    let progress_app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
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
            let _ = progress_app.emit(
                "data-directory-move-progress",
                DataDirectoryMoveProgress {
                    copied_bytes: copied,
                    total_bytes: total,
                    percent,
                },
            );
        })
    })
    .await
    .map_err(|error| error.to_string())??;
    // Relaunch instead of hot-swapping the live database/model-file handles:
    // simpler and safer than re-opening every connection in place, at the
    // cost of a brief restart the caller should warn the user about first.
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::process::Command::new(exe).spawn();
    }
    app.exit(0);
    Ok(())
}
