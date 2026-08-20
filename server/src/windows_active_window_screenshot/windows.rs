//! GDI BitBlt-based active-window capture (CPU fallback for Windows Graphics
//! Capture).

use super::encode_png_rgba;
use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;

pub fn capture_window_png(window_handle: isize) -> Result<Vec<u8>, String> {
    let hwnd = HWND(window_handle as *mut core::ffi::c_void);
    let mut rect = RECT::default();
    unsafe {
        GetClientRect(hwnd, &mut rect).map_err(|error| error.to_string())?;
    }
    let width = rect.right - rect.left;
    let height = rect.bottom - rect.top;
    if width <= 0 || height <= 0 || (width as i64 * height as i64) > 16_000_000 {
        return Err("window dimensions are invalid or exceed screenshot limit".into());
    }
    let source = unsafe { GetDC(Some(hwnd)) };
    if source.is_invalid() {
        return Err("GetDC failed".into());
    }
    let memory = unsafe { CreateCompatibleDC(Some(source)) };
    if memory.is_invalid() {
        unsafe {
            ReleaseDC(Some(hwnd), source);
        }
        return Err("CreateCompatibleDC failed".into());
    }
    let bitmap = unsafe { CreateCompatibleBitmap(source, width, height) };
    if bitmap.is_invalid() {
        unsafe {
            let _ = DeleteDC(memory);
            let _ = ReleaseDC(Some(hwnd), source);
        }
        return Err("CreateCompatibleBitmap failed".into());
    }
    let previous = unsafe { SelectObject(memory, bitmap.into()) };
    let copied =
        unsafe { BitBlt(memory, 0, 0, width, height, Some(source), 0, 0, SRCCOPY).is_ok() };
    let mut pixels = vec![0u8; width as usize * height as usize * 4];
    let mut info = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: core::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width,
            biHeight: -height,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let read = if copied {
        unsafe {
            GetDIBits(
                memory,
                bitmap,
                0,
                height as u32,
                Some(pixels.as_mut_ptr().cast()),
                &mut info,
                DIB_RGB_COLORS,
            )
        }
    } else {
        0
    };
    unsafe {
        SelectObject(memory, previous);
        let _ = DeleteObject(bitmap.into());
        let _ = DeleteDC(memory);
        let _ = ReleaseDC(Some(hwnd), source);
    }
    if read == 0 {
        return Err("GetDIBits failed".into());
    }
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.swap(0, 2);
        pixel[3] = 255;
    }
    encode_png_rgba(width as u32, height as u32, &pixels)
}

#[cfg(test)]
#[path = "../tests/windows_active_window_screenshot_windows_tests.rs"]
mod tests;
