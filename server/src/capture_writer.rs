//! Single writer thread for all capture sources.
//!
//! Mouse, keyboard, foreground-window, and filesystem capture never touch
//! SQLite directly — they send normalized `RawEvent`s down a channel to this
//! thread, which batches them into one transaction every ~200ms (or sooner if
//! a batch fills up). This keeps capture threads (especially the low-level
//! input hook threads, which must stay responsive to avoid stalling system
//! input) from ever blocking on a database lock or an fsync.

use crate::activity_capture::CaptureSettings;
use crate::local_sqlite_event_database::{Database, RawEvent};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Upper bound on how many events accumulate before a forced flush, keeping
/// a single transaction (and its memory) bounded even under a capture burst.
const MAX_BATCH_SIZE: usize = 128;
const FLUSH_INTERVAL: Duration = Duration::from_millis(200);

/// Millisecond epoch timestamp of the most recent user input (a mouse click
/// or keypress, not movement). The AI processing worker reads this to back
/// off briefly during active use rather than competing with the user for
/// CPU/GPU on the same machine.
pub static LAST_INPUT_AT_MS: AtomicI64 = AtomicI64::new(0);

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

pub fn mark_input_activity() {
    LAST_INPUT_AT_MS.store(now_millis(), Ordering::Relaxed);
}

/// Milliseconds since the last recorded input, or `i64::MAX` if none has
/// been recorded yet this run.
pub fn millis_since_last_input() -> i64 {
    match LAST_INPUT_AT_MS.load(Ordering::Relaxed) {
        0 => i64::MAX,
        last => (now_millis() - last).max(0),
    }
}

/// Bounded, memory-only holding area for screenshot bytes captured at the
/// moment of a meaningful event (e.g. a window focus change), so the AI
/// queue worker can analyze the frame the user actually saw instead of
/// re-capturing later — by which point the window may have closed, moved,
/// or been covered. Entries are consumed (removed) once a worker picks them
/// up, and the oldest entry is evicted if the cache fills, so memory use
/// stays bounded and nothing ever touches disk.
pub struct ScreenshotCache {
    entries: std::collections::HashMap<String, Vec<u8>>,
    order: std::collections::VecDeque<String>,
    max_entries: usize,
}

impl ScreenshotCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
            max_entries: max_entries.max(1),
        }
    }

    pub fn insert(&mut self, event_id: String, bytes: Vec<u8>) {
        if !self.entries.contains_key(&event_id) {
            self.order.push_back(event_id.clone());
        }
        self.entries.insert(event_id, bytes);
        while self.entries.len() > self.max_entries {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
    }

    pub fn take(&mut self, event_id: &str) -> Option<Vec<u8>> {
        let bytes = self.entries.remove(event_id);
        if bytes.is_some() {
            self.order.retain(|id| id != event_id);
        }
        bytes
    }
}

#[cfg(test)]
#[path = "tests/capture_writer_tests.rs"]
mod tests;

pub fn start_capture_writer(
    database: Arc<Mutex<Database>>,
    settings: Arc<Mutex<CaptureSettings>>,
    receiver: Receiver<RawEvent>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        let first = match receiver.recv_timeout(FLUSH_INTERVAL) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let mut batch = Vec::with_capacity(16);
        batch.push(first);
        while batch.len() < MAX_BATCH_SIZE {
            match receiver.try_recv() {
                Ok(event) => batch.push(event),
                Err(_) => break,
            }
        }
        let screenshots_enabled = settings
            .lock()
            .map(|settings| settings.screenshots_enabled)
            .unwrap_or(false);
        match database.lock() {
            Ok(database) => {
                if let Err(error) =
                    database.insert_events_and_enqueue_batch(&batch, screenshots_enabled)
                {
                    tracing::warn!(%error, count = batch.len(), "failed to persist capture batch");
                }
            }
            Err(error) => tracing::warn!(%error, "failed to lock database for capture batch"),
        }
    })
}
