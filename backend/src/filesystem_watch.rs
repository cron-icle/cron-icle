//! User-selected filesystem activity capture.
//!
//! This provider watches only folders explicitly selected by the user and
//! records filesystem evidence rather than claiming who edited a file.
//!
//! Watching is event-driven via the `notify` crate (which uses
//! `ReadDirectoryChangesW` on Windows) instead of periodic recursive
//! rescans/diffs. The background thread only wakes to service OS filesystem
//! notifications, apply the stop signal, or reconcile the watched-folder set
//! against current settings.

use crate::capture::activity::CaptureSettings;
use crate::persistence::sqlite::{Database, RawEvent};
use chrono::Utc;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;
use uuid::Uuid;

pub fn start_filesystem_loop(
    database: Arc<Mutex<Database>>,
    stop: Arc<AtomicBool>,
    settings: Arc<Mutex<CaptureSettings>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let (sender, receiver) = channel::<notify::Result<Event>>();
        let mut watcher = match notify::recommended_watcher(move |event| {
            // notify invokes this callback from its own OS-notification
            // thread; forward events to our loop rather than doing work here.
            let _ = sender.send(event);
        }) {
            Ok(watcher) => watcher,
            Err(error) => {
                tracing::warn!(%error, "failed to create filesystem watcher; filesystem capture disabled for this session");
                // Still honor the stop signal so shutdown continues to work.
                while !stop.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(200));
                }
                return;
            }
        };

        let mut watched_folders: Vec<String> = Vec::new();
        while !stop.load(Ordering::Relaxed) {
            reconcile_watches(&mut watcher, &settings, &mut watched_folders);

            match receiver.recv_timeout(Duration::from_millis(250)) {
                Ok(Ok(event)) => handle_event(&database, &settings, &event),
                Ok(Err(error)) => {
                    tracing::warn!(%error, "filesystem watcher reported an error")
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    })
}

/// Adds/removes OS watches so they match the user's current watched-folder
/// settings. Cheap (no filesystem scanning) compared to the watch calls
/// themselves, which is why this can run every loop iteration.
fn reconcile_watches(
    watcher: &mut RecommendedWatcher,
    settings: &Arc<Mutex<CaptureSettings>>,
    watched_folders: &mut Vec<String>,
) {
    let folders = settings
        .lock()
        .map(|settings| settings.watched_folders.clone())
        .unwrap_or_default();
    if folders == *watched_folders {
        return;
    }
    for old in watched_folders.iter() {
        if !folders.contains(old) {
            if let Err(error) = watcher.unwatch(Path::new(old)) {
                tracing::warn!(%error, folder = %old, "failed to unwatch removed folder");
            }
        }
    }
    for new in &folders {
        if !watched_folders.contains(new) {
            if let Err(error) = watcher.watch(Path::new(new), RecursiveMode::Recursive) {
                tracing::warn!(%error, folder = %new, "failed to watch folder");
            }
        }
    }
    *watched_folders = folders;
}

fn handle_event(database: &Arc<Mutex<Database>>, settings: &Arc<Mutex<CaptureSettings>>, event: &Event) {
    let event_type = match &event.kind {
        EventKind::Create(_) => "file_created",
        EventKind::Modify(notify::event::ModifyKind::Name(_)) => "file_renamed",
        EventKind::Modify(_) => "file_modified",
        EventKind::Remove(_) => "file_deleted",
        _ => return,
    };
    let excluded_paths = settings
        .lock()
        .map(|settings| settings.excluded_paths.clone())
        .unwrap_or_default();
    for path in &event.paths {
        if path_is_excluded(path, &excluded_paths) {
            continue;
        }
        // Directories generate their own change notifications but Cronicle
        // records file-level evidence only.
        if path.is_dir() {
            continue;
        }
        persist(database, event_type, &path.to_string_lossy());
    }
}

/// Path-component-aware exclusion check (see `CaptureSettings::excludes_path`
/// for the same logic applied to individual event paths). Using component
/// comparison instead of raw substring search keeps a short exclusion like
/// "skip" from matching an unrelated folder such as "skipper".
fn path_is_excluded(path: &Path, excluded_paths: &[String]) -> bool {
    let candidate_components = crate::capture::activity::path_components(&path.to_string_lossy());
    excluded_paths.iter().any(|excluded| {
        let excluded = excluded.trim();
        if excluded.is_empty() {
            return false;
        }
        let excluded_components = crate::capture::activity::path_components(excluded);
        crate::capture::activity::contains_component_sequence(
            &candidate_components,
            &excluded_components,
        )
    })
}

fn persist(database: &Arc<Mutex<Database>>, event_type: &str, path: &str) {
    let event = RawEvent {
        id: Uuid::new_v4().to_string(),
        timestamp_ns: Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        event_type: event_type.into(),
        source: "filesystem_watch".into(),
        app_name: None,
        executable_path: None,
        process_id: None,
        window_handle: None,
        window_title: None,
        element_name: None,
        text: None,
        file_path: Some(path.into()),
        metadata_json: "{}".into(),
        privacy_class: "filesystem_metadata".into(),
        confidence: 1.0,
        created_at: Utc::now().to_rfc3339(),
    };
    match database.lock() {
        Ok(database) => {
            if let Err(error) = database.insert_event_and_enqueue(&event) {
                tracing::warn!(%error, path = %path, "failed to persist filesystem event");
            }
        }
        Err(error) => tracing::warn!(%error, "failed to lock database for filesystem event"),
    }
}

#[cfg(test)]
#[path = "tests/filesystem_activity_capture_tests.rs"]
mod tests;
