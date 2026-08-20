//! macOS active-window capture (not yet implemented).
//!
//! Chronicle is Windows-first today; this stub keeps the crate buildable on
//! non-Windows targets and marks where a real `CGWindowListCreateImage`-based
//! capture belongs once macOS support is picked up.

pub fn capture_window_png(_window_handle: isize) -> Result<Vec<u8>, String> {
    Err("Windows screenshot provider is unavailable on this platform".into())
}
