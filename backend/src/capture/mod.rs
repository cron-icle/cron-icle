//! Capture providers: everything that turns raw OS activity into
//! normalized events. Each submodule follows the same platform-split
//! contract: a shared `mod.rs` contract, `windows.rs` (native Win32/WinRT
//! APIs), and `portable.rs` (macOS + Linux, backed by `xcap`/`rdev`/
//! `active-win-pos-rs` — see each submodule's `portable.rs` for per-crate
//! rationale and platform caveats). `ui_automation` is the one exception:
//! semantic accessibility-tree reads have no established cross-platform
//! crate, so it stays Windows-only for now (see README's Known limitations).

pub mod active_window;
pub mod activity;
pub mod graphics_session;
pub mod input;
pub mod screenshot;
pub mod ui_automation;
