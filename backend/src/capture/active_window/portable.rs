//! macOS/Linux active-window screenshot capture via `xcap`.
//!
//! `xcap::Window` gives per-window capture on both X11 Linux and macOS
//! (Wayland support depends on the compositor's screencast portal, which
//! `xcap` uses when available — see its crate docs). `window_handle` is a
//! Windows HWND concept with no portable equivalent, so a handle of `0`
//! (the sentinel `capture/activity/portable.rs` passes for "whatever is
//! currently focused") falls back to querying `is_focused()`; a nonzero
//! handle is treated as an `xcap` window id.

use super::encode_png_rgba;

pub fn capture_window_png(window_handle: isize) -> Result<Vec<u8>, String> {
    let windows = xcap::Window::all().map_err(|error| error.to_string())?;
    let window = if window_handle != 0 {
        windows
            .into_iter()
            .find(|window| window.id().map(|id| id as isize) == Ok(window_handle))
    } else {
        windows.into_iter().find(|window| window.is_focused().unwrap_or(false))
    }
    .ok_or_else(|| "no matching window found".to_string())?;

    let image = window.capture_image().map_err(|error| error.to_string())?;
    let (width, height) = (image.width(), image.height());
    encode_png_rgba(width, height, image.as_raw())
}
