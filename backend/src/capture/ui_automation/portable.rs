//! macOS/Linux focused-element accessibility read: not yet implemented.
//!
//! A real provider needs `AXUIElement` (macOS Accessibility API) or AT-SPI2
//! (Linux) tree walking, neither of which has a well-maintained general
//! crate the way screenshot/input capture do — see README's "Known
//! limitations" for tracking. `WindowsUiAutomationProvider::is_available()`
//! already reports `false` off Windows (see `mod.rs`), so callers correctly
//! treat this as "no semantic element data available" rather than silently
//! misreporting focus state.

use super::FocusedElementSnapshot;

pub fn focused_element() -> Result<Option<FocusedElementSnapshot>, String> {
    Ok(None)
}
