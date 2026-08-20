//! Privacy-first keyboard and mouse event normalization.
//!
//! Global input hooks must be explicitly enabled by the user. This module
//! keeps the normalized event contract independent from the hook mechanism,
//! so protected-field filtering and application exclusions are applied
//! before any text or coordinates are persisted. The hook mechanism itself
//! is OS-specific and lives in `windows.rs` (real implementation) and
//! `mac.rs` (not yet implemented).

#[cfg(not(windows))]
mod mac;
#[cfg(windows)]
mod windows;

#[cfg(windows)]
pub use windows::{start_keyboard_hook, start_mouse_hook};

#[cfg(not(windows))]
pub use mac::{start_keyboard_hook, start_mouse_hook};

use crate::persistence::sqlite::RawEvent;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

pub const MIN_TEXT_BATCH_DEBOUNCE: Duration = Duration::from_millis(500);
pub const MAX_TEXT_BATCH_DEBOUNCE: Duration = Duration::from_millis(1000);

#[derive(Debug, Default)]
pub struct MetadataTextBatcher {
    buffered: String,
    last_input: Option<Instant>,
}
impl MetadataTextBatcher {
    pub fn push(&mut self, text: &str) {
        self.buffered.push_str(text);
        self.last_input = Some(Instant::now());
    }
    pub fn flush_if_due(&mut self, debounce: Duration) -> Option<String> {
        let debounce = debounce.clamp(MIN_TEXT_BATCH_DEBOUNCE, MAX_TEXT_BATCH_DEBOUNCE);
        if self
            .last_input
            .is_some_and(|last| last.elapsed() >= debounce)
        {
            self.last_input = None;
            return Some(std::mem::take(&mut self.buffered));
        }
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InputCaptureSettings {
    pub mouse_enabled: bool,
    pub keyboard_enabled: bool,
    pub capture_keyboard_text: bool,
    #[serde(default)]
    pub keyboard_text_allowlist: Vec<String>,
    pub excluded_applications: Vec<String>,
}

impl InputCaptureSettings {
    pub fn allows_keyboard_text(&self, app_name: &str) -> bool {
        self.capture_keyboard_text
            && self
                .keyboard_text_allowlist
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(app_name))
    }
}

pub fn normalize_mouse_event(
    event_type: &str,
    x: i32,
    y: i32,
    button: Option<&str>,
    app_name: Option<String>,
) -> RawEvent {
    RawEvent {
        id: Uuid::new_v4().to_string(),
        timestamp_ns: Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        event_type: event_type.into(),
        source: "mouse_hook".into(),
        app_name,
        executable_path: None,
        process_id: None,
        window_handle: None,
        window_title: None,
        element_name: None,
        text: None,
        file_path: None,
        metadata_json: serde_json::json!({ "x": x, "y": y, "button": button }).to_string(),
        privacy_class: "input_metadata".into(),
        confidence: 1.0,
        created_at: Utc::now().to_rfc3339(),
    }
}

pub fn normalize_keyboard_event(
    event_type: &str,
    key_code: u32,
    app_name: Option<String>,
    text: Option<String>,
) -> RawEvent {
    RawEvent {
        id: Uuid::new_v4().to_string(),
        timestamp_ns: Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        event_type: event_type.into(),
        source: "keyboard_hook".into(),
        app_name,
        executable_path: None,
        process_id: None,
        window_handle: None,
        window_title: None,
        element_name: None,
        text,
        file_path: None,
        metadata_json: serde_json::json!({ "key_code": key_code }).to_string(),
        privacy_class: "input_metadata".into(),
        confidence: 1.0,
        created_at: Utc::now().to_rfc3339(),
    }
}

pub fn normalize_allowlisted_keyboard_event(
    settings: &InputCaptureSettings,
    event_type: &str,
    key_code: u32,
    app_name: Option<String>,
    text: Option<String>,
) -> RawEvent {
    let allowed_text = app_name
        .as_deref()
        .filter(|name| settings.allows_keyboard_text(name))
        .and(text);
    normalize_keyboard_event(event_type, key_code, app_name, allowed_text)
}

#[cfg(test)]
#[path = "../../tests/input_capture_tests.rs"]
mod tests;
