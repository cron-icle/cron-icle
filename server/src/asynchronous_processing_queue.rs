//! Asynchronous post-capture processing contracts.
//!
//! Queue tasks are deliberately separate from capture. A slow or unavailable
//! model must not block persistence of raw evidence. Workers will claim bounded
//! batches, retry transient failures, and retain model/version metadata.

use crate::capture_writer::{millis_since_last_input, ScreenshotCache};
use crate::embedding_provider::TextEmbedder;
use crate::local_model_provider::LlamaCppProvider;
use crate::local_sqlite_event_database::Database;
use crate::local_sqlite_event_database::SemanticEvent;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;
use uuid::Uuid;

pub const MAX_RETRY_ATTEMPTS: u32 = 3;
pub const MAX_PENDING_TASKS: u32 = 10_000;
/// Keeps model memory and UI latency bounded while still amortizing HTTP/model overhead.
pub const MAX_MODEL_BATCH_SIZE: usize = 8;
/// Minimum pause after each processed batch. Local inference is CPU/GPU
/// heavy; without a floor here a full queue would drive the worker loop
/// back-to-back with zero yield, competing with the rest of the machine even
/// though each individual batch is already bounded in size.
const MIN_BATCH_PACING: Duration = Duration::from_millis(300);
/// While the user has interacted (click or keypress) more recently than
/// this, the worker steps aside instead of claiming new work, so local
/// inference never competes with the user for CPU/GPU during active use.
const RECENT_INPUT_BACKOFF_MS: i64 = 400;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ProcessingMetrics {
    pub completed: u64,
    pub failed: u64,
    pub panicked: u64,
    pub total_latency_ms: u128,
    pub last_model_name: Option<String>,
    pub last_model_version: Option<String>,
}
impl ProcessingMetrics {
    pub fn snapshot(&self) -> Self {
        self.clone()
    }
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    pub fn record_completed(&mut self) {
        self.completed += 1;
    }
    pub fn record_completed_with_latency(&mut self, latency: Duration) {
        self.record_completed();
        self.total_latency_ms += latency.as_millis();
    }
    pub fn record_failed(&mut self) {
        self.failed += 1;
    }
    pub fn record_panicked(&mut self) {
        self.panicked += 1;
    }
    pub fn record_model(&mut self, name: impl Into<String>, version: impl Into<String>) {
        self.last_model_name = Some(name.into());
        self.last_model_version = Some(version.into());
    }
    pub fn average_latency_ms(&self) -> Option<f64> {
        (self.completed > 0).then(|| self.total_latency_ms as f64 / self.completed as f64)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    SemanticTextAnalysis,
    SemanticImageAnalysis,
    EmbeddingGeneration,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QueueStatus {
    Pending,
    Processing,
    Complete,
    Failed,
    Cancelled,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueTask {
    pub id: String,
    pub raw_event_id: String,
    pub task_type: TaskType,
    pub status: QueueStatus,
    pub attempts: u32,
    pub priority: i32,
}
pub trait SemanticAnalyzer: Send + Sync {
    fn analyze_text(&self, input: &str) -> Result<String, String>;
    fn analyze_image(&self, bytes: &[u8]) -> Result<String, String>;
}
pub trait Embedder: Send + Sync {
    fn embed(&self, input: &str) -> Result<Vec<f32>, String>;
}
pub fn retry_delay(attempt: u32) -> Duration {
    Duration::from_millis(250u64.saturating_mul(2u64.saturating_pow(attempt.min(8))))
}

pub trait QueueTaskProcessor: Send + Sync {
    fn process(&self, task: &QueueTask) -> Result<(), String>;
    fn process_batch(&self, tasks: &[QueueTask]) -> Result<(), String> {
        for task in tasks {
            self.process(task)?;
        }
        Ok(())
    }
}

pub struct LocalModelQueueProcessor {
    pub database: Arc<Mutex<Database>>,
    pub screenshot_cache: Arc<Mutex<ScreenshotCache>>,
}
fn persist_semantic_result(
    database: &Arc<Mutex<Database>>,
    task: &QueueTask,
    provider: &LlamaCppProvider,
    output: crate::local_semantic_processing::SemanticModelOutput,
) -> Result<(), String> {
    if database
        .lock()
        .map_err(|_| "database lock poisoned")?
        .semantic_for_raw_event(&task.raw_event_id)
        .map_err(|e| e.to_string())?
        .is_some()
    {
        return Ok(());
    }
    database
        .lock()
        .map_err(|_| "database lock poisoned")?
        .insert_semantic_event(&SemanticEvent {
            id: Uuid::new_v4().to_string(),
            raw_event_id: task.raw_event_id.clone(),
            category: output.category,
            summary: output.summary,
            entities_json: serde_json::to_string(&output.entities).unwrap_or_default(),
            relationships_json: serde_json::to_string(&output.relationships).unwrap_or_default(),
            confidence: output.confidence,
            model_name: provider.chat_model.clone(),
            model_version: "llama.cpp".into(),
            created_at: Utc::now().to_rfc3339(),
        })
        .map_err(|e| e.to_string())?;
    database
        .lock()
        .map_err(|_| "database lock poisoned")?
        .enqueue_task(&QueueTask {
            id: Uuid::new_v4().to_string(),
            raw_event_id: task.raw_event_id.clone(),
            task_type: TaskType::EmbeddingGeneration,
            status: QueueStatus::Pending,
            attempts: 0,
            priority: -1,
        })
        .map_err(|e| e.to_string())
}
impl QueueTaskProcessor for LocalModelQueueProcessor {
    fn process(&self, task: &QueueTask) -> Result<(), String> {
        let provider = LlamaCppProvider::default();
        let database = self.database.clone();
        let event = database
            .lock()
            .map_err(|_| "database lock poisoned")?
            .event_by_id(&task.raw_event_id)
            .map_err(|e| e.to_string())?
            .ok_or("raw event not found")?;
        let context = format!(
            "application: {:?}\nwindow: {:?}\nevent: {}\ntext: {:?}",
            event.app_name, event.window_title, event.event_type, event.text
        );
        match task.task_type {
            TaskType::SemanticTextAnalysis => {
                let output = provider.analyze_text(&context)?;
                persist_semantic_result(&database, task, &provider, output)
            }
            TaskType::EmbeddingGeneration => {
                let embedding = provider.embed(&context)?;
                let semantic_id = database
                    .lock()
                    .map_err(|_| "database lock poisoned")?
                    .semantic_for_raw_event(&task.raw_event_id)
                    .map_err(|e| e.to_string())?
                    .ok_or("semantic event not found")?
                    .id;
                database
                    .lock()
                    .map_err(|_| "database lock poisoned")?
                    .insert_embedding(&semantic_id, &provider.embedding_model, "llama.cpp", &embedding)
                    .map_err(|e| e.to_string())
            }
            TaskType::SemanticImageAnalysis => {
                // Prefer the frame captured at event time (the window was
                // guaranteed foregrounded then); only fall back to a live
                // capture if that memory-only cache entry is gone, e.g. after
                // an app restart or once the entry has already been consumed.
                let cached = self
                    .screenshot_cache
                    .lock()
                    .ok()
                    .and_then(|mut cache| cache.take(&task.raw_event_id));
                let image = match cached {
                    Some(bytes) => bytes,
                    None => {
                        let window_handle = event
                            .window_handle
                            .ok_or("image task has no window handle")?;
                        crate::windows_graphics_capture_session::capture_one_frame_png(
                            window_handle as isize,
                        )
                        .or_else(|_| {
                            crate::windows_active_window_screenshot::capture_window_png(
                                window_handle as isize,
                            )
                        })?
                    }
                };
                let output = provider.analyze_image(&image)?;
                persist_semantic_result(&database, task, &provider, output)
            }
        }
    }

    fn process_batch(&self, tasks: &[QueueTask]) -> Result<(), String> {
        if tasks.len() <= 1
            || tasks
                .iter()
                .any(|task| task.task_type == TaskType::SemanticImageAnalysis)
        {
            return self.process(tasks.first().ok_or("empty processing batch")?);
        }
        let provider = LlamaCppProvider::default();
        let database = self.database.clone();
        let contexts = tasks
            .iter()
            .map(|task| {
                database
                    .lock()
                    .map_err(|_| "database lock poisoned".to_string())?
                    .event_by_id(&task.raw_event_id)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| "raw event not found".to_string())
                    .map(|event| {
                        format!(
                            "application: {:?}\nwindow: {:?}\nevent: {}\ntext: {:?}",
                            event.app_name, event.window_title, event.event_type, event.text
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        match tasks[0].task_type {
            TaskType::SemanticTextAnalysis => {
                let outputs = provider.analyze_text_batch(&contexts).or_else(|_| {
                    contexts
                        .iter()
                        .map(|context| provider.analyze_text(context))
                        .collect()
                })?;
                for (task, output) in tasks.iter().zip(outputs) {
                    persist_semantic_result(&database, task, &provider, output)?;
                }
            }
            TaskType::EmbeddingGeneration => {
                let embeddings = provider.embed_batch(&contexts).or_else(|_| {
                    contexts
                        .iter()
                        .map(|context| provider.embed(context))
                        .collect()
                })?;
                for (task, embedding) in tasks.iter().zip(embeddings) {
                    let semantic_id = database
                        .lock()
                        .map_err(|_| "database lock poisoned")?
                        .semantic_for_raw_event(&task.raw_event_id)
                        .map_err(|e| e.to_string())?
                        .ok_or("semantic event not found")?
                        .id;
                    database
                        .lock()
                        .map_err(|_| "database lock poisoned")?
                        .insert_embedding(&semantic_id, &provider.embedding_model, "llama.cpp", &embedding)
                        .map_err(|e| e.to_string())?;
                }
            }
            TaskType::SemanticImageAnalysis => unreachable!(),
        }
        Ok(())
    }
}

pub fn run_processing_worker(
    database: Arc<Mutex<Database>>,
    stop: Arc<AtomicBool>,
    processor: Arc<dyn QueueTaskProcessor>,
) -> thread::JoinHandle<()> {
    run_processing_worker_with_metrics(database, stop, processor, Arc::new(Mutex::new(ProcessingMetrics::default())))
}

/// Same worker loop as `run_processing_worker`, but records throughput,
/// failure counts, and per-batch latency into `metrics` as it goes — the
/// only way to answer "is the pipeline actually keeping up" from outside
/// the worker thread. `run_processing_worker` exists as a thin wrapper for
/// callers (like most tests) that don't need to observe this; production
/// startup (`start_capture_state`) uses this directly and hands the same
/// `Arc<Mutex<ProcessingMetrics>>` to the `processing_metrics` Tauri command.
pub fn run_processing_worker_with_metrics(
    database: Arc<Mutex<Database>>,
    stop: Arc<AtomicBool>,
    processor: Arc<dyn QueueTaskProcessor>,
    metrics: Arc<Mutex<ProcessingMetrics>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        if let Ok(database) = database.lock() {
            let _ = database.recover_stale_processing_tasks(10);
        }
        while !stop.load(Ordering::Relaxed) {
            // Step aside while the user is actively clicking/typing so local
            // inference never competes with them for CPU/GPU on the same
            // machine; capture keeps running uninterrupted (it never goes
            // through this worker), only new AI work waits.
            if millis_since_last_input() < RECENT_INPUT_BACKOFF_MS {
                thread::sleep(Duration::from_millis(RECENT_INPUT_BACKOFF_MS as u64));
                continue;
            }
            let task = database
                .lock()
                .ok()
                .and_then(|database| database.claim_next_task().ok())
                .flatten();
            let Some(task) = task else {
                // No work right now is exactly when it's safe (and useful)
                // to check whether the generation engine has sat idle past
                // its keep-alive and should give its memory back — see
                // `native_inference::sweep_idle_engines`.
                crate::native_inference::sweep_idle_engines();
                thread::sleep(Duration::from_millis(250));
                continue;
            };
            if stop.load(Ordering::Relaxed) {
                if let Ok(database) = database.lock() {
                    let _ = database.requeue_processing_tasks();
                }
                break;
            }
            let mut tasks = vec![task];
            let task_type = tasks[0].task_type.clone();
            if task_type != TaskType::SemanticImageAnalysis {
                // Scale the batch down from MAX_MODEL_BATCH_SIZE when
                // available RAM is tight rather than always claiming a
                // fixed 8 — each additional item in a text-analysis batch
                // adds to the same prompt/context, and an embedding batch
                // adds another sequence to the same context, so a memory-
                // constrained machine benefits from smaller batches the
                // same way it benefits from a smaller context size.
                let batch_size = crate::memory_planner::adaptive_batch_size(
                    MAX_MODEL_BATCH_SIZE,
                    &crate::hardware_profiler::HardwareProfile::detect(),
                );
                if let Ok(database) = database.lock() {
                    if let Ok(mut additional) =
                        database.claim_next_tasks(&task_type, batch_size.saturating_sub(1))
                    {
                        tasks.append(&mut additional);
                    }
                }
            }
            let batch_started = std::time::Instant::now();
            let panicked = std::cell::Cell::new(false);
            let processing_result =
                catch_unwind(AssertUnwindSafe(|| processor.process_batch(&tasks))).unwrap_or_else(|_| {
                    panicked.set(true);
                    Err("processing provider panicked".into())
                });
            let batch_latency = batch_started.elapsed();
            match processing_result {
                Ok(()) => {
                    if let Ok(database) = database.lock() {
                        for task in &tasks {
                            let _ = database.finish_task(&task.id);
                        }
                    }
                    if let Ok(mut metrics) = metrics.lock() {
                        for _ in &tasks {
                            metrics.record_completed_with_latency(batch_latency);
                        }
                    }
                }
                Err(error) => {
                    if let Ok(database) = database.lock() {
                        for task in &tasks {
                            let retry = task.attempts < MAX_RETRY_ATTEMPTS;
                            let _ = database.fail_task(&task.id, &error, retry, task.attempts);
                        }
                    }
                    if let Ok(mut metrics) = metrics.lock() {
                        for _ in &tasks {
                            if panicked.get() {
                                metrics.record_panicked();
                            } else {
                                metrics.record_failed();
                            }
                        }
                    }
                    if tasks.iter().any(|task| task.attempts < MAX_RETRY_ATTEMPTS) {
                        thread::sleep(retry_delay(tasks[0].attempts));
                    }
                }
            }
            // Bounded pacing: even with a full queue, never drive the model
            // back-to-back at 100% — a short yield between batches keeps
            // memory/CPU/GPU use predictable instead of spiking for as long
            // as backlog exists.
            thread::sleep(MIN_BATCH_PACING);
        }
        if let Ok(database) = database.lock() {
            let _ = database.requeue_processing_tasks();
        }
    })
}
#[cfg(test)]
#[path = "tests/asynchronous_processing_queue_tests.rs"]
mod tests;
