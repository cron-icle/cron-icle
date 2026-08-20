//! Windows Graphics Capture D3D11 session initialization.
//!
//! The session owns the GPU capture resources and reads frames back to PNG.
//! This is Windows-only end to end (WinRT + D3D11), so there is no shared
//! code here — just the platform dispatch. `mac.rs` is a stub until macOS
//! support (ScreenCaptureKit) is picked up.

#[cfg(not(windows))]
mod mac;
#[cfg(windows)]
mod windows;

#[cfg(windows)]
#[allow(unused_imports)] // D3d11CaptureSession is part of the public return-type contract
pub use windows::{capture_one_frame_png, initialize, D3d11CaptureSession};

#[cfg(not(windows))]
pub use mac::{capture_one_frame_png, initialize};
