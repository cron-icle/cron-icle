//! macOS active-window screenshot provider (not yet implemented).
//!
//! Chronicle is Windows-first today; this stub keeps the crate buildable on
//! non-Windows targets and marks where a real provider belongs once macOS
//! support is picked up.

use super::ActiveWindowScreenshotProvider;

pub struct MacActiveWindowScreenshotProvider {
    pub window_handle: isize,
}

impl ActiveWindowScreenshotProvider for MacActiveWindowScreenshotProvider {
    fn capture_active_window(&self) -> Result<Vec<u8>, String> {
        crate::windows_active_window_screenshot::capture_window_png(self.window_handle)
    }
}

pub fn graphics_capture_item_available(_window_handle: isize) -> bool {
    false
}
