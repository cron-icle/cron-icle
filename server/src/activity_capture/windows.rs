//! Windows foreground activity provider entry point.
//!
//! Foreground tracking is event-driven via `SetWinEventHook`
//! (`EVENT_SYSTEM_FOREGROUND`, `WINEVENT_OUTOFCONTEXT`) rather than polling
//! `GetForegroundWindow` on a timer. `WINEVENT_OUTOFCONTEXT` hooks deliver
//! their callback on the thread that registered the hook via that thread's
//! message queue, so this module also runs a small message pump on the
//! capture thread. The pump idles briefly between messages purely to stay
//! responsive to the stop signal — window-change detection itself is driven
//! entirely by the OS event, not by re-polling window state.

use super::CaptureSettings;
use crate::local_sqlite_event_database::Database;
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

pub const PROVIDER_NAME: &str = "windows_foreground_activity";

struct HookContext {
    database: Arc<Mutex<Database>>,
    settings: Arc<Mutex<CaptureSettings>>,
    screenshot_cache: Arc<Mutex<crate::capture_writer::ScreenshotCache>>,
    previous: Option<(isize, String)>,
}

thread_local! {
    // `SetWinEventHook(..., WINEVENT_OUTOFCONTEXT)` always invokes the
    // callback on the thread that registered the hook, so thread-local
    // storage is a safe way to hand capture state to the extern "C" callback
    // without global mutable state.
    static HOOK_CONTEXT: RefCell<Option<HookContext>> = RefCell::new(None);
}

/// Registers the foreground-window hook and runs the message pump until
/// `stop` is set. Falls back to a stop-signal-only wait loop (no window
/// tracking) if the hook cannot be installed, so capture threads still shut
/// down cleanly.
pub fn run_foreground_hook_loop(
    database: Arc<Mutex<Database>>,
    stop: Arc<AtomicBool>,
    settings: Arc<Mutex<CaptureSettings>>,
    screenshot_cache: Arc<Mutex<crate::capture_writer::ScreenshotCache>>,
) {
    use ::windows::Win32::Foundation::HWND;
    use ::windows::Win32::UI::Accessibility::{SetWinEventHook, UnhookWinEvent};
    use ::windows::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, PeekMessageW, TranslateMessage, EVENT_SYSTEM_FOREGROUND, MSG,
        PM_REMOVE, WINEVENT_OUTOFCONTEXT,
    };

    HOOK_CONTEXT.with(|context| {
        *context.borrow_mut() = Some(HookContext {
            database,
            settings,
            screenshot_cache,
            previous: None,
        });
    });

    let hook = unsafe {
        SetWinEventHook(
            EVENT_SYSTEM_FOREGROUND,
            EVENT_SYSTEM_FOREGROUND,
            None,
            Some(win_event_proc),
            0,
            0,
            WINEVENT_OUTOFCONTEXT,
        )
    };

    if hook.is_invalid() {
        tracing::warn!("SetWinEventHook failed; foreground window tracking disabled for this session");
        while !stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(200));
        }
        HOOK_CONTEXT.with(|context| *context.borrow_mut() = None);
        return;
    }

    // Prime state with whatever is currently focused so the first real
    // change is detected against a known baseline.
    process_foreground_change();

    while !stop.load(Ordering::Relaxed) {
        let mut message = MSG::default();
        unsafe {
            // A non-blocking pump: dispatches any pending hook callback
            // messages immediately (event-driven), and otherwise sleeps
            // briefly only so this thread can notice the stop signal without
            // an indefinite blocking wait.
            if PeekMessageW(&mut message, Some(HWND(std::ptr::null_mut())), 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            } else {
                thread::sleep(Duration::from_millis(50));
            }
        }
    }

    unsafe {
        let _ = UnhookWinEvent(hook);
    }
    HOOK_CONTEXT.with(|context| *context.borrow_mut() = None);
}

unsafe extern "system" fn win_event_proc(
    _hook: ::windows::Win32::UI::Accessibility::HWINEVENTHOOK,
    event: u32,
    _hwnd: ::windows::Win32::Foundation::HWND,
    id_object: i32,
    id_child: i32,
    _event_thread: u32,
    _event_time: u32,
) {
    use ::windows::Win32::UI::WindowsAndMessaging::EVENT_SYSTEM_FOREGROUND;
    // OBJID_WINDOW (0) / CHILDID_SELF (0) filters out control-level noise so
    // only whole-window foreground changes are handled.
    if event == EVENT_SYSTEM_FOREGROUND && id_object == 0 && id_child == 0 {
        process_foreground_change();
    }
}

fn process_foreground_change() {
    HOOK_CONTEXT.with(|context| {
        let mut borrowed = context.borrow_mut();
        let Some(context) = borrowed.as_mut() else {
            return;
        };
        let Some((handle, title, process_id, executable_path, app_name)) =
            super::current_foreground_window()
        else {
            return;
        };
        let excluded = context
            .settings
            .lock()
            .map(|settings| settings.excludes_application(&executable_path, &app_name))
            .unwrap_or(false);
        if excluded {
            return;
        }
        let changed = context
            .previous
            .as_ref()
            .map(|(old_handle, old_title)| *old_handle != handle || old_title != &title)
            .unwrap_or(true);
        if !changed {
            return;
        }
        let event_type = if context
            .previous
            .as_ref()
            .map(|(old_handle, _)| *old_handle == handle)
            .unwrap_or(false)
        {
            "window_title_changed"
        } else {
            "window_focused"
        };
        let mut event = super::normalize_window_event(
            app_name,
            title.clone(),
            Some(executable_path),
            Some(process_id),
        );
        event.window_handle = Some(handle as u64);
        event.event_type = event_type.into();
        event.metadata_json = format!("{{\"window_handle\":{handle}}}");
        // A window-handle event with screenshots enabled is enqueued for
        // image analysis (see `insert_event_and_enqueue`), so the frame is
        // captured here — while the window is guaranteed to still be
        // foregrounded — and handed to the queue worker via the shared
        // cache, rather than the worker re-capturing later when the window
        // may have moved, minimized, or closed.
        let screenshots_enabled = context
            .settings
            .lock()
            .map(|settings| settings.screenshots_enabled)
            .unwrap_or(false);
        if screenshots_enabled {
            let captured = crate::windows_graphics_capture_session::capture_one_frame_png(handle)
                .or_else(|_| crate::windows_active_window_screenshot::capture_window_png(handle));
            match captured {
                Ok(bytes) => {
                    if let Ok(mut cache) = context.screenshot_cache.lock() {
                        cache.insert(event.id.clone(), bytes);
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, event_id = %event.id, "failed to capture screenshot for foreground event");
                }
            }
        }
        match context.database.lock() {
            Ok(database) => {
                if let Err(error) = database.insert_event_and_enqueue(&event) {
                    tracing::warn!(%error, event_id = %event.id, "failed to persist foreground event");
                }
            }
            Err(error) => {
                tracing::warn!(%error, "failed to lock database for foreground event")
            }
        }
        context.previous = Some((handle, title));
    });
}
