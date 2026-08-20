//! COM/UI Automation focused-element read.

use super::FocusedElementSnapshot;

pub fn focused_element() -> Result<Option<FocusedElementSnapshot>, String> {
    use windows::core::PCWSTR;
    use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};
    use windows::Win32::UI::Accessibility::CUIAutomation;
    use windows::Win32::UI::Shell::SHCoCreateInstance;
    let initialized = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.is_ok();
    if !initialized {
        return Ok(None);
    }
    struct ComGuard;
    impl Drop for ComGuard {
        fn drop(&mut self) {
            unsafe {
                CoUninitialize();
            }
        }
    }
    let _com_guard = ComGuard;
    let automation: windows::Win32::UI::Accessibility::IUIAutomation =
        unsafe { SHCoCreateInstance(PCWSTR::null(), Some(&CUIAutomation), None) }
            .map_err(|error| error.to_string())?;
    let element =
        unsafe { automation.GetFocusedElement() }.map_err(|error| error.to_string())?;
    let name = unsafe { element.CurrentName() }
        .ok()
        .map(|value| value.to_string());
    let automation_id = unsafe { element.CurrentAutomationId() }
        .ok()
        .map(|value| value.to_string());
    let class_name = unsafe { element.CurrentClassName() }
        .ok()
        .map(|value| value.to_string());
    let framework_id = unsafe { element.CurrentFrameworkId() }
        .ok()
        .map(|value| value.to_string());
    let control_type = unsafe { element.CurrentLocalizedControlType() }
        .ok()
        .map(|value| value.to_string());
    let bounds = unsafe { element.CurrentBoundingRectangle() }.ok().map(|value| serde_json::json!({ "left": value.left, "top": value.top, "right": value.right, "bottom": value.bottom }).to_string());
    let selected_text = unsafe { read_selected_text(&element) };
    Ok(Some(
        FocusedElementSnapshot {
            automation_id,
            control_type,
            element_name: name,
            element_value: None,
            class_name,
            framework_id,
            bounds_json: bounds,
            selected_text,
        }
        .sanitize(),
    ))
}

unsafe fn read_selected_text(
    element: &windows::Win32::UI::Accessibility::IUIAutomationElement,
) -> Option<String> {
    use windows::core::Interface;
    use windows::Win32::System::Ole::{
        SafeArrayDestroy, SafeArrayGetElement, SafeArrayGetLBound, SafeArrayGetUBound,
    };
    use windows::Win32::UI::Accessibility::{ITextProvider, ITextRangeProvider, UIA_TextPatternId};
    let provider: ITextProvider = element.GetCurrentPatternAs(UIA_TextPatternId).ok()?;
    let array = provider.GetSelection().ok()?;
    if array.is_null() {
        return None;
    }
    let lower = SafeArrayGetLBound(array, 1).ok()?;
    let upper = SafeArrayGetUBound(array, 1).ok()?;
    let mut selected = None;
    for index in lower..=upper {
        let mut raw: *mut core::ffi::c_void = core::ptr::null_mut();
        if SafeArrayGetElement(array, &index, &mut raw as *mut _ as *mut core::ffi::c_void).is_ok()
            && !raw.is_null()
        {
            let range = ITextRangeProvider::from_raw(raw);
            if let Ok(text) = range.GetText(4096) {
                let value = text.to_string();
                if !value.trim().is_empty() {
                    selected = Some(value);
                    break;
                }
            }
        }
    }
    let _ = SafeArrayDestroy(array);
    selected
}
