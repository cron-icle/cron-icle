//! Windows low-level mouse/keyboard hook boundary.
//!
//! The OS callback is intentionally tiny: it copies the event data into a
//! channel and immediately returns. Mouse movement is never captured — only
//! discrete clicks, scroll, and key presses reach the channel, so ordinary
//! use never floods the pipeline. The hook thread's job is to pump Windows
//! messages (required for the low-level hook to keep receiving callbacks)
//! and forward drained messages to the shared capture-writer channel; it
//! never touches the database itself, so a slow database write can never
//! stall the message pump and delay input for the rest of the system.

use super::{normalize_keyboard_event, normalize_mouse_event};
use crate::capture_writer::mark_input_activity;
use crate::local_sqlite_event_database::RawEvent;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

/// How often the hook thread checks for a drained message when idle. Small
/// enough to keep `pump_window_messages` running frequently (the low-level
/// hook mechanism depends on this thread's message queue staying serviced),
/// large enough to avoid a busy spin.
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(8);

#[derive(Clone, Copy)]
struct MouseMessage {
    event_type: &'static str,
    x: i32,
    y: i32,
    button: Option<&'static str>,
}

// Write-once outer cell, mutable inner slot: lets a hook be stopped and
// later restarted (settings toggled off/on) without the second `start_*`
// call silently failing to register a sender, which is what happened when
// this used `OnceLock::set` directly.
static MOUSE_SENDER: OnceLock<Mutex<Option<mpsc::Sender<MouseMessage>>>> = OnceLock::new();

static KEYBOARD_SENDER: OnceLock<Mutex<Option<mpsc::Sender<u32>>>> = OnceLock::new();

pub fn start_keyboard_hook(
    writer: mpsc::Sender<RawEvent>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let (sender, receiver) = mpsc::channel();
        *KEYBOARD_SENDER.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(sender);
        let hook = unsafe { install_keyboard_hook() };
        while !stop.load(Ordering::Relaxed) {
            pump_window_messages();
            match receiver.try_recv() {
                Ok(key_code) => {
                    mark_input_activity();
                    let event = normalize_keyboard_event("key_down", key_code, None, None);
                    let _ = writer.send(event);
                }
                Err(mpsc::TryRecvError::Empty) => thread::sleep(IDLE_POLL_INTERVAL),
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        if !hook.is_invalid() {
            unsafe {
                let _ = windows::Win32::UI::WindowsAndMessaging::UnhookWindowsHookEx(hook);
            }
        }
        if let Some(slot) = KEYBOARD_SENDER.get() {
            if let Ok(mut sender) = slot.lock() {
                *sender = None;
            }
        }
    })
}

unsafe fn install_keyboard_hook() -> windows::Win32::UI::WindowsAndMessaging::HHOOK {
    use windows::Win32::UI::WindowsAndMessaging::{SetWindowsHookExW, WH_KEYBOARD_LL};
    SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_callback), None, 0).unwrap_or_default()
}

unsafe extern "system" fn keyboard_callback(
    code: i32,
    wparam: windows::Win32::Foundation::WPARAM,
    lparam: windows::Win32::Foundation::LPARAM,
) -> windows::Win32::Foundation::LRESULT {
    use windows::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, KBDLLHOOKSTRUCT, WM_KEYDOWN, WM_SYSKEYDOWN,
    };
    if code >= 0 && lparam.0 != 0 && matches!(wparam.0 as u32, WM_KEYDOWN | WM_SYSKEYDOWN) {
        let data = &*(lparam.0 as *const KBDLLHOOKSTRUCT);
        if let Some(slot) = KEYBOARD_SENDER.get() {
            if let Ok(sender) = slot.lock() {
                if let Some(sender) = sender.as_ref() {
                    let _ = sender.send(data.vkCode);
                }
            }
        }
    }
    CallNextHookEx(None, code, wparam, lparam)
}

pub fn start_mouse_hook(
    writer: mpsc::Sender<RawEvent>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let (sender, receiver) = mpsc::channel();
        *MOUSE_SENDER.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(sender);
        let hook = unsafe { install_hook() };
        let mut last_left_click: Option<(Instant, i32, i32)> = None;
        while !stop.load(Ordering::Relaxed) {
            pump_window_messages();
            match receiver.try_recv() {
                Ok(message) => {
                    mark_input_activity();
                    let mut event_type = message.event_type;
                    if message.event_type == "mouse_click" {
                        if let Some((time, old_x, old_y)) = last_left_click {
                            if time.elapsed() <= Duration::from_millis(500)
                                && (old_x - message.x).abs() <= 4
                                && (old_y - message.y).abs() <= 4
                            {
                                event_type = "mouse_double_click";
                            }
                        }
                        last_left_click = Some((Instant::now(), message.x, message.y));
                    }
                    let event = normalize_mouse_event(
                        event_type,
                        message.x,
                        message.y,
                        message.button,
                        None,
                    );
                    let _ = writer.send(event);
                }
                Err(mpsc::TryRecvError::Empty) => thread::sleep(IDLE_POLL_INTERVAL),
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        if !hook.is_invalid() {
            unsafe {
                let _ = windows::Win32::UI::WindowsAndMessaging::UnhookWindowsHookEx(hook);
            }
        }
        if let Some(slot) = MOUSE_SENDER.get() {
            if let Ok(mut sender) = slot.lock() {
                *sender = None;
            }
        }
    })
}

fn pump_window_messages() {
    use windows::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
    };
    let mut message = MSG::default();
    unsafe {
        while PeekMessageW(&mut message, None, 0, 0, PM_REMOVE).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

unsafe fn install_hook() -> windows::Win32::UI::WindowsAndMessaging::HHOOK {
    use windows::Win32::UI::WindowsAndMessaging::{SetWindowsHookExW, WH_MOUSE_LL};
    SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_callback), None, 0).unwrap_or_default()
}

/// Only discrete clicks and scroll are ever sent to the channel.
/// `WM_MOUSEMOVE` (and button-up, which carries no new information once the
/// down-click has already been recorded) are acknowledged to Windows via
/// `CallNextHookEx` but never leave the callback — mouse movement must never
/// become a persisted event or an AI queue task.
unsafe extern "system" fn mouse_callback(
    code: i32,
    wparam: windows::Win32::Foundation::WPARAM,
    lparam: windows::Win32::Foundation::LPARAM,
) -> windows::Win32::Foundation::LRESULT {
    use windows::Win32::UI::WindowsAndMessaging::CallNextHookEx;
    use windows::Win32::UI::WindowsAndMessaging::{
        MSLLHOOKSTRUCT, WM_LBUTTONDOWN, WM_MBUTTONDOWN, WM_MOUSEWHEEL, WM_RBUTTONDOWN,
    };
    if code >= 0 && lparam.0 != 0 {
        let data = &*(lparam.0 as *const MSLLHOOKSTRUCT);
        let message = match wparam.0 as u32 {
            WM_LBUTTONDOWN => Some(("mouse_click", Some("left"))),
            WM_RBUTTONDOWN => Some(("mouse_right_click", Some("right"))),
            WM_MBUTTONDOWN => Some(("mouse_click", Some("middle"))),
            WM_MOUSEWHEEL => Some(("mouse_scroll", None)),
            _ => None,
        };
        if let Some((event_type, button)) = message {
            if let Some(slot) = MOUSE_SENDER.get() {
                if let Ok(sender) = slot.lock() {
                    if let Some(sender) = sender.as_ref() {
                        let _ = sender.send(MouseMessage {
                            event_type,
                            x: data.pt.x,
                            y: data.pt.y,
                            button,
                        });
                    }
                }
            }
        }
    }
    CallNextHookEx(None, code, wparam, lparam)
}
