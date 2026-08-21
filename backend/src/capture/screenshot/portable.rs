//! macOS/Linux active-window screenshot provider, backed by `xcap`
//! (see `capture/active_window/portable.rs` for the actual capture call).

use super::ActiveWindowScreenshotProvider;

pub struct PortableActiveWindowScreenshotProvider {
    pub window_handle: isize,
}

impl ActiveWindowScreenshotProvider for PortableActiveWindowScreenshotProvider {
    fn capture_active_window(&self) -> Result<Vec<u8>, String> {
        crate::capture::active_window::capture_window_png(self.window_handle)
    }
}

/// Windows' probe checks whether the OS can hand back a
/// `GraphicsCaptureItem` for this window without actually capturing.
/// `xcap` has no equivalent cheap check, so this reports whether the window
/// is currently enumerable at all (a reasonable proxy for "capturable").
pub fn graphics_capture_item_available(window_handle: isize) -> bool {
    let Ok(windows) = xcap::Window::all() else {
        return false;
    };
    if window_handle == 0 {
        return windows.iter().any(|window| window.is_focused().unwrap_or(false));
    }
    windows
        .iter()
        .any(|window| window.id().map(|id| id as isize) == Ok(window_handle))
}
