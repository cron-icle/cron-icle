//! Capture providers: everything that turns raw OS activity into
//! normalized events. Each submodule follows the same platform-split
//! contract (a shared `mod.rs`, plus `windows.rs`/`mac.rs`).

pub mod active_window;
pub mod activity;
pub mod graphics_session;
pub mod input;
pub mod screenshot;
pub mod ui_automation;
