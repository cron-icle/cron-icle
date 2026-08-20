//! Windows UI Automation focused-element capture boundary.
//!
//! UI Automation metadata is preferred over screenshots because it can expose
//! semantic control information without retaining visual assets. The provider
//! must treat missing patterns and inaccessible/elevated applications as
//! normal outcomes, not capture failures. The COM/UIA integration is
//! Windows-only (`windows.rs`); `mac.rs` is a stub until an Accessibility-API
//! based provider is implemented.

#[cfg(not(windows))]
mod mac;
#[cfg(windows)]
mod windows;

use crate::local_sqlite_event_database::RawEvent;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FocusedElementSnapshot {
    pub automation_id: Option<String>,
    pub control_type: Option<String>,
    pub element_name: Option<String>,
    pub element_value: Option<String>,
    pub class_name: Option<String>,
    pub framework_id: Option<String>,
    pub bounds_json: Option<String>,
    pub selected_text: Option<String>,
}

impl FocusedElementSnapshot {
    pub fn sanitize(mut self) -> Self {
        self.element_name = self
            .element_name
            .map(|value| value.chars().take(512).collect());
        self.element_value = self
            .element_value
            .map(|value| value.chars().take(4096).collect());
        self.selected_text = self
            .selected_text
            .map(|value| value.chars().take(4096).collect());
        if self
            .control_type
            .as_deref()
            .is_some_and(|value| value.to_ascii_lowercase().contains("password"))
        {
            self.element_value = None;
            self.selected_text = None;
        }
        self
    }
}

pub trait UiAutomationProvider: Send {
    fn is_available(&self) -> bool;
    fn focused_element(&self) -> Result<Option<FocusedElementSnapshot>, String>;
}

pub struct WindowsUiAutomationProvider;
impl UiAutomationProvider for WindowsUiAutomationProvider {
    fn is_available(&self) -> bool {
        cfg!(windows)
    }
    fn focused_element(&self) -> Result<Option<FocusedElementSnapshot>, String> {
        #[cfg(windows)]
        {
            windows::focused_element()
        }
        #[cfg(not(windows))]
        {
            mac::focused_element()
        }
    }
}

pub fn normalize_focused_element(snapshot: FocusedElementSnapshot) -> RawEvent {
    let snapshot = snapshot.sanitize();
    let protected = snapshot
        .control_type
        .as_deref()
        .is_some_and(|value| value.to_ascii_lowercase().contains("password"));
    RawEvent { id: Uuid::new_v4().to_string(), timestamp_ns: Utc::now().timestamp_nanos_opt().unwrap_or_default(), event_type: "element_focused".into(), source: "windows_ui_automation".into(), app_name: None, executable_path: None, process_id: None, window_handle: None, window_title: None, element_name: snapshot.element_name, text: snapshot.selected_text, file_path: None, metadata_json: serde_json::json!({ "automation_id": snapshot.automation_id, "control_type": snapshot.control_type, "element_value": snapshot.element_value, "class_name": snapshot.class_name, "framework_id": snapshot.framework_id, "bounds": snapshot.bounds_json }).to_string(), privacy_class: if protected { "protected_field".into() } else { "ui_automation_metadata".into() }, confidence: 1.0, created_at: Utc::now().to_rfc3339() }
}

#[cfg(test)]
#[path = "../tests/windows_ui_automation_capture_tests.rs"]
mod tests;
