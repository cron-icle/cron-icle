//! Windows active-window screenshot provider and capture-item probe.

use super::ActiveWindowScreenshotProvider;

pub struct WindowsActiveWindowScreenshotProvider {
    pub window_handle: isize,
}

impl ActiveWindowScreenshotProvider for WindowsActiveWindowScreenshotProvider {
    fn capture_active_window(&self) -> Result<Vec<u8>, String> {
        crate::windows_active_window_screenshot::capture_window_png(self.window_handle)
    }
}

pub fn graphics_capture_item_available(window_handle: isize) -> bool {
    use windows::Graphics::Capture::GraphicsCaptureItem;
    use windows::UI::WindowId;
    GraphicsCaptureItem::TryCreateFromWindowId(WindowId {
        Value: window_handle as u64,
    })
    .is_ok()
}
