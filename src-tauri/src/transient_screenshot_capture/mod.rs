//! Transient screenshot capture and asset lifecycle.
//!
//! Screenshot bytes are intentionally not part of `RawEvent` or SQLite. This
//! module owns short-lived in-memory assets that may be handed to an analysis
//! queue and are then dropped, including when analysis fails. The active-
//! window capture call itself is OS-specific: the real implementation is in
//! `windows.rs`, with `mac.rs` reserved for a future macOS provider.

#[cfg(not(windows))]
mod mac;
#[cfg(windows)]
mod windows;

#[cfg(windows)]
#[allow(unused_imports)] // exercised by the graphics_capture_probe_is_safe_for_invalid_handle test
pub use windows::{graphics_capture_item_available, WindowsActiveWindowScreenshotProvider as PlatformActiveWindowScreenshotProvider};

#[cfg(not(windows))]
#[allow(unused_imports)]
pub use mac::{graphics_capture_item_available, MacActiveWindowScreenshotProvider as PlatformActiveWindowScreenshotProvider};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const DEFAULT_SCREENSHOT_RETENTION: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScreenshotTrigger {
    AppActivated,
    WindowTitleChanged,
    DoubleClick,
    RightClick,
    TextSelected,
    DragEnded,
    ElementFocused,
}

impl ScreenshotTrigger {
    pub fn meaningful(self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct TransientScreenshotAsset {
    pub raw_event_id: String,
    pub queue_task_id: Option<String>,
    pub captured_at: Instant,
    pub bytes: Vec<u8>,
    pub mime_type: String,
}

impl TransientScreenshotAsset {
    pub fn new(raw_event_id: String, bytes: Vec<u8>, mime_type: impl Into<String>) -> Self {
        Self {
            raw_event_id,
            queue_task_id: None,
            captured_at: Instant::now(),
            bytes,
            mime_type: mime_type.into(),
        }
    }
    pub fn expired(&self, retention: Duration) -> bool {
        self.captured_at.elapsed() >= retention
    }
    pub fn is_valid(&self) -> bool {
        !self.bytes.is_empty() && self.mime_type.starts_with("image/")
    }
}

#[derive(Default)]
pub struct TransientScreenshotStore {
    assets: HashMap<String, TransientScreenshotAsset>,
}

#[derive(Debug, Default)]
pub struct ScreenshotTriggerDispatcher {
    pending: Vec<(String, ScreenshotTrigger)>,
}
impl ScreenshotTriggerDispatcher {
    pub fn request(&mut self, raw_event_id: impl Into<String>, trigger: ScreenshotTrigger) {
        if trigger.meaningful() {
            self.pending.push((raw_event_id.into(), trigger));
        }
    }
    pub fn drain(&mut self) -> Vec<(String, ScreenshotTrigger)> {
        std::mem::take(&mut self.pending)
    }
}

impl TransientScreenshotStore {
    pub fn insert(&mut self, asset: TransientScreenshotAsset) -> bool {
        if !asset.is_valid() {
            return false;
        }
        self.assets.insert(asset.raw_event_id.clone(), asset);
        true
    }
    pub fn associate_queue_task(&mut self, raw_event_id: &str, queue_task_id: String) -> bool {
        if let Some(asset) = self.assets.get_mut(raw_event_id) {
            asset.queue_task_id = Some(queue_task_id);
            true
        } else {
            false
        }
    }
    pub fn take(&mut self, raw_event_id: &str) -> Option<TransientScreenshotAsset> {
        self.assets.remove(raw_event_id)
    }
    pub fn purge_expired(&mut self, retention: Duration) {
        self.assets.retain(|_, asset| !asset.expired(retention));
    }
    pub fn purge_default_retention(&mut self) {
        self.purge_expired(DEFAULT_SCREENSHOT_RETENTION);
    }
    pub fn len(&self) -> usize {
        self.assets.len()
    }
}

pub trait ActiveWindowScreenshotProvider: Send {
    fn capture_active_window(&self) -> Result<Vec<u8>, String>;
}

#[cfg(test)]
#[path = "../tests/transient_screenshot_capture_tests.rs"]
mod tests;
