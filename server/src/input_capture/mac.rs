//! macOS mouse/keyboard hook boundary (not yet implemented).
//!
//! Chronicle is Windows-first today; this stub keeps the crate buildable on
//! non-Windows targets and marks where a real `CGEventTap`-based provider
//! belongs once macOS support is picked up. Threads exit immediately since
//! there is no hook to pump messages for.

use crate::local_sqlite_event_database::RawEvent;
use std::sync::{atomic::AtomicBool, mpsc, Arc};
use std::thread;

pub fn start_keyboard_hook(
    _writer: mpsc::Sender<RawEvent>,
    _stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(|| {})
}

pub fn start_mouse_hook(
    _writer: mpsc::Sender<RawEvent>,
    _stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(|| {})
}
