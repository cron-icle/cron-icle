//! One-time local AI setup, driven from inside the app.
//!
//! Chronicle's semantic analysis and embeddings run on a bundled llama.cpp
//! engine (`llama-server`) rather than a separately installed application:
//! nothing here shows up in the Start Menu, the system tray, or Windows'
//! installed-apps list. The `llama-server` binary itself ships alongside the
//! Chronicle binary (see `backend/resources/llama` and
//! `model_provider::engine_paths::runtime_dir`) — nothing here
//! downloads it. This module only downloads the GGUF model files (Gemma 3
//! for chat/vision, EmbeddingGemma for embeddings, both from their official
//! Hugging Face repos) into `<data dir>\llama\models` (see
//! `data_directory::current`, the folder the user chooses from Settings —
//! the download commands below refuse to run until one is chosen), starts/
//! stops the two local servers, and removes downloaded model files again on
//! request. Every step is UI-triggered and streams real, byte-accurate
//! progress into `AppState::download_progress`, which the UI polls (also
//! mirrored to `tracing`, so the same information is visible in the
//! terminal and the browser UI). Nothing here runs automatically or
//! silently in the background.

use crate::inference::model_provider::{engine_paths, shared_agent, LlamaCppProvider};
use crate::state::AppState;
use serde::Serialize;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct LlamaSetupStatus {
    pub runtime_installed: bool,
    pub chat_model_installed: bool,
    pub embed_model_installed: bool,
    pub chat_running: bool,
    pub embed_running: bool,
    pub chat_model_name: String,
    pub embed_model_name: String,
}

fn setup_status_blocking() -> LlamaSetupStatus {
    let engine = LlamaCppProvider::default();
    LlamaSetupStatus {
        runtime_installed: engine_paths::runtime_installed(),
        chat_model_installed: engine_paths::chat_model_installed(),
        embed_model_installed: engine_paths::embed_model_installed(),
        chat_running: engine.chat_reachable(),
        embed_running: engine.embed_reachable(),
        chat_model_name: engine.chat_model.clone(),
        embed_model_name: engine.embedding_model.clone(),
    }
}

pub async fn local_ai_setup_status() -> LlamaSetupStatus {
    tokio::task::spawn_blocking(setup_status_blocking)
        .await
        .unwrap_or(LlamaSetupStatus {
            runtime_installed: false,
            chat_model_installed: false,
            embed_model_installed: false,
            chat_running: false,
            embed_running: false,
            chat_model_name: String::new(),
            embed_model_name: String::new(),
        })
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub label: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub percent: Option<f32>,
}

fn set_download_progress(
    state: &AppState,
    label: &str,
    downloaded: u64,
    total: Option<u64>,
    force: bool,
    last_emit: &mut Instant,
) {
    if !force && last_emit.elapsed() < Duration::from_millis(200) {
        return;
    }
    *last_emit = Instant::now();
    let percent = total
        .filter(|&total| total > 0)
        .map(|total| (downloaded as f64 / total as f64 * 100.0) as f32);
    if let Ok(mut slot) = state.download_progress.lock() {
        *slot = Some(DownloadProgress {
            label: label.to_string(),
            downloaded_bytes: downloaded,
            total_bytes: total,
            percent,
        });
    }
}

/// Current progress of an in-flight model download, if any. Polled by the
/// UI instead of pushed, since there is no event channel once the app runs
/// as a plain HTTP server.
pub fn download_progress(state: &AppState) -> Option<DownloadProgress> {
    state.download_progress.lock().ok().and_then(|guard| guard.clone())
}

/// Streams `url` to `dest`, byte by byte, recording real (not estimated)
/// progress from the response's `Content-Length` header into
/// `AppState::download_progress`. Writes to a `.part` sibling file and
/// renames on success, so a failed/cancelled download never leaves a file
/// that looks complete but isn't.
///
/// Checks `cancel` after every chunk (at most 64KiB of latency, not one long
/// unresponsive read) and, if set, stops and deletes the partial `.part`
/// file rather than leaving a truncated download that looks resumable but
/// isn't — the next attempt starts clean from byte zero.
fn download_with_progress(
    state: &AppState,
    label: &str,
    url: &str,
    dest: &Path,
    cancel: &AtomicBool,
) -> Result<(), String> {
    tracing::info!(target: "chronicle::local_inference_setup", "downloading {label} from {url}");
    let response = shared_agent()
        .get(url)
        .call()
        .map_err(|error| format!("failed to start download of {label}: {error}"))?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let mut reader = response.into_body().into_reader();
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let tmp_path = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp_path)
        .map_err(|error| format!("failed to create {}: {error}", tmp_path.display()))?;
    let mut buffer = [0u8; 65536];
    let mut downloaded: u64 = 0;
    let mut last_emit = Instant::now() - Duration::from_secs(1);
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(&tmp_path);
            return Err(format!("{label} download was cancelled"));
        }
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("download of {label} was interrupted: {error}"))?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .map_err(|error| format!("failed writing {label}: {error}"))?;
        downloaded += read as u64;
        set_download_progress(state, label, downloaded, total, false, &mut last_emit);
    }
    set_download_progress(state, label, downloaded, total, true, &mut last_emit);
    drop(file);
    std::fs::rename(&tmp_path, dest)
        .map_err(|error| format!("failed to finalize {label}: {error}"))?;
    tracing::info!(target: "chronicle::local_inference_setup", "{label} downloaded ({downloaded} bytes)");
    Ok(())
}

const NO_DATA_DIRECTORY_ERROR: &str =
    "Choose a data directory in Settings before downloading local AI models.";

/// Requests the in-flight model download, if any, stop as soon as it next
/// checks (see `download_with_progress`) — cooperative, not instantaneous,
/// since the underlying HTTP read can't be interrupted mid-syscall.
pub fn cancel_model_download(state: &AppState) -> Result<(), String> {
    state.download_cancel.store(true, Ordering::Relaxed);
    Ok(())
}

/// Downloads the Gemma 3 chat/vision model and its multimodal projector.
/// Runs synchronously; callers that want a non-blocking HTTP response
/// (`http_api`) spawn this on a background task and let the caller poll
/// `download_progress` instead of awaiting completion.
pub async fn setup_download_chat_model(state: Arc<AppState>) -> Result<(), String> {
    state.download_cancel.store(false, Ordering::Relaxed);
    let cancel = state.download_cancel.clone();
    let state_for_blocking = state.clone();
    tokio::task::spawn_blocking(move || {
        let (Some(chat_model), Some(mmproj)) = (engine_paths::chat_model(), engine_paths::mmproj())
        else {
            return Err(NO_DATA_DIRECTORY_ERROR.to_string());
        };
        download_with_progress(&state_for_blocking, "Gemma 3 chat model", engine_paths::CHAT_MODEL_URL, &chat_model, &cancel)?;
        download_with_progress(&state_for_blocking, "Gemma 3 vision projector", engine_paths::MMPROJ_URL, &mmproj, &cancel)
    })
    .await
    .map_err(|error| error.to_string())?
}

/// Downloads the EmbeddingGemma model. See `setup_download_chat_model` for
/// the non-blocking-caller convention.
pub async fn setup_download_embed_model(state: Arc<AppState>) -> Result<(), String> {
    state.download_cancel.store(false, Ordering::Relaxed);
    let cancel = state.download_cancel.clone();
    let state_for_blocking = state.clone();
    tokio::task::spawn_blocking(move || {
        let Some(embed_model) = engine_paths::embed_model() else {
            return Err(NO_DATA_DIRECTORY_ERROR.to_string());
        };
        download_with_progress(&state_for_blocking, "EmbeddingGemma model", engine_paths::EMBED_MODEL_URL, &embed_model, &cancel)
    })
    .await
    .map_err(|error| error.to_string())?
}

/// Starts both local servers (chat/vision and embedding) if their files are
/// present and they aren't already listening, registering any child this
/// call starts in `AppState` so it's stopped the same way as one started at
/// application launch (see `shutdown_llama_engine`).
pub async fn setup_start_engine(state: &AppState) -> Result<(), String> {
    let engine = LlamaCppProvider::default();
    let chat_engine = engine.clone();
    let chat_child = tokio::task::spawn_blocking(move || chat_engine.start_chat_server_if_needed())
        .await
        .map_err(|error| error.to_string())??;
    if let Some(child) = chat_child {
        if let Ok(mut slot) = state.llama_chat_process.lock() {
            if slot.is_none() {
                *slot = Some(child);
            }
        }
    }
    let embed_engine = engine.clone();
    let embed_child = tokio::task::spawn_blocking(move || embed_engine.start_embed_server_if_needed())
        .await
        .map_err(|error| error.to_string())??;
    if let Some(child) = embed_child {
        if let Ok(mut slot) = state.llama_embed_process.lock() {
            if slot.is_none() {
                *slot = Some(child);
            }
        }
    }
    Ok(())
}

fn stop_process(slot: &std::sync::Mutex<Option<std::process::Child>>) {
    if let Ok(mut process_slot) = slot.lock() {
        if let Some(mut process) = process_slot.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

fn remove_file_if_exists(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to remove {}: {error}", path.display())),
    }
}

/// Removes the Gemma 3 chat/vision model and its projector. Stops the chat
/// server first: on Windows a running server keeps its model files locked,
/// so deleting them out from under it would fail.
pub async fn setup_remove_chat_model(state: &AppState) -> Result<(), String> {
    stop_process(&state.llama_chat_process);
    tracing::info!(target: "chronicle::local_inference_setup", "removing Gemma 3 chat model");
    tokio::task::spawn_blocking(|| {
        if let Some(chat_model) = engine_paths::chat_model() {
            remove_file_if_exists(&chat_model)?;
        }
        if let Some(mmproj) = engine_paths::mmproj() {
            remove_file_if_exists(&mmproj)?;
        }
        Ok(())
    })
    .await
    .map_err(|error| error.to_string())?
}

/// Removes the EmbeddingGemma model. Stops the embedding server first for
/// the same file-locking reason as `setup_remove_chat_model`.
pub async fn setup_remove_embed_model(state: &AppState) -> Result<(), String> {
    stop_process(&state.llama_embed_process);
    tracing::info!(target: "chronicle::local_inference_setup", "removing EmbeddingGemma model");
    tokio::task::spawn_blocking(|| match engine_paths::embed_model() {
        Some(embed_model) => remove_file_if_exists(&embed_model),
        None => Ok(()),
    })
    .await
    .map_err(|error| error.to_string())?
}
