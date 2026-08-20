//! macOS/Linux (X11) global input hook via `rdev`.
//!
//! `rdev::listen` blocks the calling thread for as long as the OS event tap
//! stays alive and offers no cooperative-stop API, unlike the Windows
//! low-level-hook loop in `windows.rs` which pumps messages and can check
//! `stop` between callbacks. So this thread checks `stop` *inside* the
//! event callback (skipping forwarding once set) rather than around a
//! message-pump loop; the OS-level listener itself is left running until
//! process exit, which is the safely-detached shutdown case AGENTS.md's
//! "background threads need a stop signal" rule allows for (there is
//! nothing left to join once the process is exiting anyway, and no data
//! keeps flowing once `stop` is observed).
//!
//! Mouse movement is never captured — only discrete clicks and scroll, to
//! match the Windows implementation's event volume.
//!
//! macOS caveat: the host process needs Accessibility permission granted
//! (System Settings > Privacy & Security > Accessibility) or `rdev` will
//! silently receive no events. Linux caveat: `listen` uses X11 APIs and
//! will not receive events under Wayland.

use super::{normalize_keyboard_event, normalize_mouse_event};
use crate::persistence::sqlite::RawEvent;
use crate::persistence::writer::mark_input_activity;
use rdev::{Event, EventType};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

pub fn start_keyboard_hook(
    writer: mpsc::Sender<RawEvent>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let callback = move |event: Event| {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            if let EventType::KeyPress(key) = event.event_type {
                mark_input_activity();
                let key_code = format!("{key:?}")
                    .bytes()
                    .fold(0u32, |acc, byte| acc.wrapping_mul(31).wrapping_add(byte as u32));
                let event = normalize_keyboard_event("key_down", key_code, None, None);
                let _ = writer.send(event);
            }
        };
        if let Err(error) = rdev::listen(callback) {
            tracing::warn!(?error, "rdev keyboard listener failed to start");
        }
    })
}

pub fn start_mouse_hook(
    writer: mpsc::Sender<RawEvent>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut last_position = (0i32, 0i32);
        let callback = move |event: Event| {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            match event.event_type {
                EventType::ButtonPress(button) => {
                    mark_input_activity();
                    let button = match button {
                        rdev::Button::Left => Some("left"),
                        rdev::Button::Right => Some("right"),
                        rdev::Button::Middle => Some("middle"),
                        _ => None,
                    };
                    let event = normalize_mouse_event(
                        "mouse_click",
                        last_position.0,
                        last_position.1,
                        button,
                        None,
                    );
                    let _ = writer.send(event);
                }
                EventType::Wheel { delta_x: _, delta_y } => {
                    mark_input_activity();
                    let event = normalize_mouse_event(
                        "mouse_scroll",
                        last_position.0,
                        last_position.1,
                        Some(if delta_y >= 0 { "up" } else { "down" }),
                        None,
                    );
                    let _ = writer.send(event);
                }
                EventType::MouseMove { x, y } => {
                    last_position = (x as i32, y as i32);
                }
                _ => {}
            }
        };
        if let Err(error) = rdev::listen(callback) {
            tracing::warn!(?error, "rdev mouse listener failed to start");
        }
    })
}
