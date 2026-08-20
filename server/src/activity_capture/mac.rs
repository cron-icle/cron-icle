//! macOS foreground-window activity provider (not yet implemented).
//!
//! Chronicle is Windows-first today; this stub keeps the crate buildable on
//! non-Windows targets and marks where a real `NSWorkspace`/Accessibility-API
//! provider belongs once macOS support is picked up.

use crate::activity_capture::CaptureSettings;
use crate::capture_writer::ScreenshotCache;
use crate::local_sqlite_event_database::RawEvent;
use std::sync::{atomic::AtomicBool, mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

pub fn start_foreground_loop(
    _writer: mpsc::Sender<RawEvent>,
    stop: Arc<AtomicBool>,
    _settings: Arc<Mutex<CaptureSettings>>,
    _screenshot_cache: Arc<Mutex<ScreenshotCache>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(500));
        }
    })
}
