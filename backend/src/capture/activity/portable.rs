//! macOS/Linux foreground-window activity provider.
//!
//! Unlike Windows' event-driven `SetWinEventHook`, neither `active-win-pos-rs`
//! (used here for the active-window query) nor a portable OS event exposes a
//! push-based "foreground changed" notification across macOS and X11 Linux
//! alike, so this polls the active window at a short fixed interval instead.
//! The poll interval is short enough that a title/app change is detected
//! well within a second, and idle polling is cheap (a single OS call).
//!
//! On Wayland, `active-win-pos-rs` cannot query the active window (no
//! portable Wayland protocol exposes it), so `get_active_window()` returns
//! `Err` and this loop degrades to "no window tracked" without erroring —
//! the stop signal is still honored so shutdown remains clean.

use super::CaptureSettings;
use crate::persistence::sqlite::Database;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

pub const PROVIDER_NAME: &str = "portable_foreground_activity";

const POLL_INTERVAL: Duration = Duration::from_millis(300);

pub fn run_foreground_poll_loop(
    database: Arc<Mutex<Database>>,
    stop: Arc<AtomicBool>,
    settings: Arc<Mutex<CaptureSettings>>,
    screenshot_cache: Arc<Mutex<crate::persistence::writer::ScreenshotCache>>,
) {
    let mut previous: Option<(String, String)> = None;
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(POLL_INTERVAL);
        let Ok(window) = active_win_pos_rs::get_active_window() else {
            continue;
        };
        let executable_path = window.process_path.to_string_lossy().to_string();
        let excluded = settings
            .lock()
            .map(|settings| settings.excludes_application(&executable_path, &window.app_name))
            .unwrap_or(false);
        if excluded {
            continue;
        }
        let changed = previous
            .as_ref()
            .map(|(old_id, old_title)| *old_id != window.window_id || old_title != &window.title)
            .unwrap_or(true);
        if !changed {
            continue;
        }
        let event_type = if previous
            .as_ref()
            .map(|(old_id, _)| *old_id == window.window_id)
            .unwrap_or(false)
        {
            "window_title_changed"
        } else {
            "window_focused"
        };
        let mut event = super::normalize_window_event(
            window.app_name.clone(),
            window.title.clone(),
            Some(executable_path),
            Some(window.process_id as u32),
        );
        event.event_type = event_type.into();
        event.metadata_json = format!("{{\"window_id\":{:?}}}", window.window_id);

        let screenshots_enabled = settings
            .lock()
            .map(|settings| settings.screenshots_enabled)
            .unwrap_or(false);
        if screenshots_enabled {
            match crate::capture::active_window::capture_window_png(0) {
                Ok(bytes) => {
                    if let Ok(mut cache) = screenshot_cache.lock() {
                        cache.insert(event.id.clone(), bytes);
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, event_id = %event.id, "failed to capture screenshot for foreground event");
                }
            }
        }
        match database.lock() {
            Ok(database) => {
                if let Err(error) = database.insert_event_and_enqueue(&event) {
                    tracing::warn!(%error, event_id = %event.id, "failed to persist foreground event");
                }
            }
            Err(error) => {
                tracing::warn!(%error, "failed to lock database for foreground event")
            }
        }
        previous = Some((window.window_id, window.title));
    }
}
