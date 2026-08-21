//! Windows activity capture providers.
//!
//! This module owns the boundary between OS activity APIs and Cronicle's
//! normalized raw-event model. Capture runs on a background thread and must
//! never wait for semantic AI processing. Everything in this file is
//! platform-independent (settings, normalization, exclusion matching,
//! provider contract, and thread lifecycle); the actual OS integration lives
//! in `windows.rs`, with `mac.rs` reserved for a future macOS provider.

#[cfg(windows)]
mod windows;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod portable;

use crate::persistence::sqlite::RawEvent;
use chrono::Utc;
use serde::{Deserialize, Serialize};
#[cfg_attr(windows, allow(unused_imports))]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
#[cfg_attr(windows, allow(unused_imports))]
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CaptureSettings {
    pub enabled: bool,
    pub mouse_enabled: bool,
    pub keyboard_enabled: bool,
    pub keyboard_mode: KeyboardMode,
    #[serde(default)]
    pub keyboard_text_allowlist: Vec<String>,
    pub excluded_applications: Vec<String>,
    #[serde(default)]
    pub excluded_paths: Vec<String>,
    pub watched_folders: Vec<String>,
    pub screenshots_enabled: bool,
}

impl CaptureSettings {
    /// Matches on exact executable filename/stem (case-insensitive), never
    /// on a raw substring. This deliberately avoids over-matching, e.g. an
    /// exclusion of "code" must match "Code.exe" but must NOT match
    /// "decode.exe" or "Encoder.exe".
    pub fn excludes_application(&self, executable_path: &str, app_name: &str) -> bool {
        let exe_filename = executable_filename(executable_path);
        let exe_stem = executable_stem(executable_path);
        let app_stem = executable_stem(app_name);
        self.excluded_applications.iter().any(|excluded| {
            let pattern = excluded.trim();
            if pattern.is_empty() {
                return false;
            }
            let pattern_filename = executable_filename(pattern);
            let pattern_stem = executable_stem(pattern);
            (!exe_filename.is_empty() && exe_filename == pattern_filename)
                || (!exe_stem.is_empty() && exe_stem == pattern_stem)
                || (!app_stem.is_empty() && app_stem == pattern_stem)
        })
    }

    /// Matches on path-component containment rather than raw substring
    /// search, so an exclusion of "secrets" matches the folder segment
    /// `...\Secrets\...` but a fragment like "ret" cannot accidentally
    /// match an unrelated folder such as "Secretariat".
    pub fn excludes_path(&self, path: &str) -> bool {
        let candidate_components = path_components(path);
        self.excluded_paths.iter().any(|excluded| {
            let excluded = excluded.trim();
            if excluded.is_empty() {
                return false;
            }
            let excluded_components = path_components(excluded);
            contains_component_sequence(&candidate_components, &excluded_components)
        })
    }

    pub fn allows_keyboard_text(&self, app_name: &str) -> bool {
        matches!(self.keyboard_mode, KeyboardMode::AllowlistedText)
            && self
                .keyboard_text_allowlist
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(app_name))
    }
}

fn executable_filename(value: &str) -> String {
    std::path::Path::new(value.trim())
        .file_name()
        .map(|name| name.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

fn executable_stem(value: &str) -> String {
    std::path::Path::new(value.trim())
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

/// Path split into lowercase, non-empty components for segment-aware
/// comparison (as opposed to raw substring search).
pub(crate) fn path_components(value: &str) -> Vec<String> {
    std::path::Path::new(value.trim())
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_ascii_lowercase())
        .filter(|component| !component.is_empty())
        .collect()
}

/// True when `needle` appears as a contiguous run of path components inside
/// `haystack`.
pub(crate) fn contains_component_sequence(haystack: &[String], needle: &[String]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|window| window == needle)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum KeyboardMode {
    #[default]
    MetadataOnly,
    AllowlistedText,
    FullText,
}

pub trait CaptureProvider: Send {
    fn name(&self) -> &'static str;
    fn start(&mut self) -> Result<(), String>;
    fn stop(&mut self);
    fn is_available(&self) -> bool;
}

pub struct ForegroundWindowProvider {
    running: bool,
}
impl ForegroundWindowProvider {
    pub fn new() -> Self {
        Self { running: false }
    }
}
impl CaptureProvider for ForegroundWindowProvider {
    fn name(&self) -> &'static str {
        "foreground_window"
    }
    fn start(&mut self) -> Result<(), String> {
        self.running = true;
        Ok(())
    }
    fn stop(&mut self) {
        self.running = false;
    }
    fn is_available(&self) -> bool {
        cfg!(windows)
    }
}

pub fn normalize_window_event(
    app_name: String,
    window_title: String,
    executable_path: Option<String>,
    process_id: Option<u32>,
) -> RawEvent {
    RawEvent {
        id: Uuid::new_v4().to_string(),
        timestamp_ns: Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        event_type: "window_focused".into(),
        source: "foreground_window".into(),
        app_name: Some(app_name),
        executable_path,
        process_id,
        window_handle: None,
        window_title: Some(window_title),
        element_name: None,
        text: None,
        file_path: None,
        metadata_json: "{}".into(),
        privacy_class: "metadata".into(),
        confidence: 1.0,
        created_at: Utc::now().to_rfc3339(),
    }
}

/// Starts event-driven foreground-window tracking via `SetWinEventHook`
/// (see `windows::run_foreground_hook_loop`) instead of polling
/// `GetForegroundWindow` on a timer.
#[cfg(windows)]
pub fn start_foreground_loop(
    database: Arc<std::sync::Mutex<crate::persistence::sqlite::Database>>,
    stop: Arc<AtomicBool>,
    settings: Arc<std::sync::Mutex<CaptureSettings>>,
    screenshot_cache: Arc<std::sync::Mutex<crate::persistence::writer::ScreenshotCache>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        windows::run_foreground_hook_loop(database, stop, settings, screenshot_cache);
    })
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub fn start_foreground_loop(
    database: Arc<std::sync::Mutex<crate::persistence::sqlite::Database>>,
    stop: Arc<AtomicBool>,
    settings: Arc<std::sync::Mutex<CaptureSettings>>,
    screenshot_cache: Arc<std::sync::Mutex<crate::persistence::writer::ScreenshotCache>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        portable::run_foreground_poll_loop(database, stop, settings, screenshot_cache);
    })
}

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
pub fn start_foreground_loop(
    _database: Arc<std::sync::Mutex<crate::persistence::sqlite::Database>>,
    stop: Arc<AtomicBool>,
    _settings: Arc<std::sync::Mutex<CaptureSettings>>,
    _screenshot_cache: Arc<std::sync::Mutex<crate::persistence::writer::ScreenshotCache>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(500));
        }
    })
}

#[cfg(windows)]
pub(crate) fn current_foreground_window() -> Option<(isize, String, u32, String, String)> {
    use ::windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
    };
    let window = unsafe { GetForegroundWindow() };
    if window.0.is_null() {
        return None;
    }
    let length = unsafe { GetWindowTextLengthW(window) };
    let mut buffer = vec![0u16; (length + 1) as usize];
    let written = unsafe { GetWindowTextW(window, &mut buffer) };
    let title = String::from_utf16_lossy(&buffer[..written as usize]);
    let mut process_id = 0u32;
    unsafe {
        GetWindowThreadProcessId(window, Some(&mut process_id));
    }
    let executable_path = process_executable_path(process_id).unwrap_or_default();
    let app_name = executable_path
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or("unknown")
        .to_string();
    Some((
        window.0 as isize,
        title,
        process_id,
        executable_path,
        app_name,
    ))
}

#[cfg(windows)]
fn process_executable_path(process_id: u32) -> Option<String> {
    use ::windows::Win32::Foundation::CloseHandle;
    use ::windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_FORMAT,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    let process =
        unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, process_id).ok()? };
    let mut buffer = vec![0u16; 1024];
    let mut length = buffer.len() as u32;
    let success = unsafe {
        QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_FORMAT(0),
            ::windows::core::PWSTR(buffer.as_mut_ptr()),
            &mut length,
        )
        .is_ok()
    };
    unsafe {
        let _ = CloseHandle(process);
    }
    success.then(|| String::from_utf16_lossy(&buffer[..length as usize]))
}

#[cfg(test)]
#[path = "../../tests/activity_capture_tests.rs"]
mod tests;
