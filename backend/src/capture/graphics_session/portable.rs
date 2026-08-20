//! macOS/Linux "graphics session" shim.
//!
//! There is no portable equivalent of Windows Graphics Capture's persistent
//! GPU session — `xcap` (used here and in `active_window/portable.rs`) does
//! a fresh CPU-side capture per call, with no session object to hold open.
//! `initialize` therefore just proves a window is capturable right now
//! (used by the `/graphics-capture-available` probe) and `PortableCaptureSession`
//! carries nothing; `capture_one_frame_png` delegates straight to the same
//! `xcap`-backed path as `active_window::capture_window_png`.

pub struct PortableCaptureSession {
    window_handle: isize,
}

pub fn initialize(window_handle: isize) -> Result<PortableCaptureSession, String> {
    // Cheap existence probe: try a real capture once. xcap has no
    // "can I capture this" query short of actually capturing.
    crate::capture::active_window::capture_window_png(window_handle)?;
    Ok(PortableCaptureSession { window_handle })
}

pub fn capture_one_frame_png(window_handle: isize) -> Result<Vec<u8>, String> {
    crate::capture::active_window::capture_window_png(window_handle)
}

impl PortableCaptureSession {
    pub fn capture_next_frame_png(&self) -> Result<Vec<u8>, String> {
        crate::capture::active_window::capture_window_png(self.window_handle)
    }
}
