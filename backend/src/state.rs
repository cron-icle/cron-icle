//! Shared application state: database handles, capture lifecycle handles,
//! and the live progress/metrics fields the HTTP layer polls.

use crate::capture::activity::CaptureSettings;
use crate::inference::model_provider::LlamaCppProvider;
use crate::persistence::sqlite::{self, Database, ReaderPool};
use crate::processing::queue::ProcessingMetrics;
use std::process::Child;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

pub struct AppState {
    pub database: Arc<Mutex<Database>>,
    /// Read-only connection pool for query commands (list/search/count/
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
    pub screenshot_cache: Arc<Mutex<crate::persistence::writer::ScreenshotCache>>,
    /// Set to request the in-flight model download (if any) stop at its next
    /// checked point. Reset to `false` at the start of each new download.
    pub download_cancel: Arc<AtomicBool>,
    /// Throughput/latency/failure counters for the AI processing worker,
    /// updated live as it runs and surfaced to the UI so pipeline health is
    /// observable instead of only inferable from queue counts after the
    /// fact.
    pub processing_metrics: Arc<Mutex<ProcessingMetrics>>,
    /// Live progress of an in-flight local-AI model download, polled by the
    /// UI (there is no push/event channel — the app is a plain HTTP
    /// server). `None` when no download is in flight.
    pub download_progress: Mutex<Option<crate::inference::setup::DownloadProgress>>,
    /// Live progress of an in-flight data-directory move, polled the same
    /// way as `download_progress`.
    pub data_dir_move_progress: Mutex<Option<DataDirectoryMoveProgress>>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DataDirectoryMoveProgress {
    pub copied_bytes: u64,
    pub total_bytes: u64,
    pub percent: f32,
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
                        // persistence rather than crash the process.
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
            match sqlite::open_reader_pool() {
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
            screenshot_cache: Arc::new(Mutex::new(crate::persistence::writer::ScreenshotCache::new(32))),
            download_cancel: Arc::new(AtomicBool::new(false)),
            processing_metrics: Arc::new(Mutex::new(ProcessingMetrics::default())),
            download_progress: Mutex::new(None),
            data_dir_move_progress: Mutex::new(None),
        }
    }

    /// Runs a read-only query off the async executor thread. Prefers a
    /// pooled read-only connection (see `ReaderPool`) so queries never
    /// contend with the single writer mutex used by capture and processing;
    /// falls back to the writer connection when no pool is available (e.g.
    /// degraded in-memory startup). Always runs inside `spawn_blocking` so
    /// the async handler, and therefore the HTTP server, is never blocked on
    /// rusqlite I/O.
    pub(crate) async fn read_with<T, F>(&self, query: F) -> Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce(&rusqlite::Connection) -> rusqlite::Result<T> + Send + 'static,
    {
        if let Some(pool) = self.reader_pool.clone() {
            tokio::task::spawn_blocking(move || {
                let connection = pool.get().map_err(|error| error.to_string())?;
                query(&connection).map_err(|error| error.to_string())
            })
            .await
            .map_err(|error| error.to_string())?
        } else {
            let database = self.database.clone();
            tokio::task::spawn_blocking(move || {
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
