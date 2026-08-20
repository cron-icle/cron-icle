//! macOS Accessibility-API focused-element read (not yet implemented).
//!
//! Chronicle is Windows-first today; this stub keeps the crate buildable on
//! non-Windows targets and marks where a real `AXUIElement`-based provider
//! belongs once macOS support is picked up.

use super::FocusedElementSnapshot;

pub fn focused_element() -> Result<Option<FocusedElementSnapshot>, String> {
    Ok(None)
}
