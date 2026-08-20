//! macOS graphics capture session (not yet implemented).
//!
//! Chronicle is Windows-first today; this stub keeps the crate buildable on
//! non-Windows targets and marks where a real ScreenCaptureKit-based session
//! belongs once macOS support is picked up.

pub fn initialize(_window_handle: isize) -> Result<(), String> {
    Err("D3D11 capture is only available on Windows".into())
}

pub fn capture_one_frame_png(_window_handle: isize) -> Result<Vec<u8>, String> {
    Err("D3D11 capture is only available on Windows".into())
}
