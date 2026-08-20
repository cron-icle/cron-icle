//! Active-window screenshot capture and PNG encoding.
//!
//! `capture_window_png` is a CPU-backed fallback for environments where
//! Windows Graphics Capture cannot initialize. Bytes remain in memory and
//! are returned as a bounded PNG payload for the transient screenshot
//! pipeline. The GDI capture itself is Windows-only (`windows.rs`); the PNG
//! encoder below is pure and shared by both the GDI path and the D3D11 path
//! in `windows_graphics_capture_session`.

#[cfg(windows)]
mod windows;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod portable;

#[cfg(windows)]
pub use windows::capture_window_png;

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub use portable::capture_window_png;

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
pub fn capture_window_png(_window_handle: isize) -> Result<Vec<u8>, String> {
    Err("active-window screenshot capture is not implemented on this platform".into())
}

pub(crate) fn encode_png_rgba(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    if rgba.len() != width as usize * height as usize * 4 {
        return Err("pixel buffer dimensions do not match".into());
    }
    let mut scanlines = Vec::with_capacity((rgba.len() + height as usize) + 6);
    for row in rgba.chunks_exact(width as usize * 4) {
        scanlines.push(0);
        scanlines.extend_from_slice(row);
    }
    let mut compressed = Vec::with_capacity(scanlines.len() + 6 + scanlines.len() / 65535 * 5);
    compressed.extend_from_slice(&[0x78, 0x01]);
    for (index, chunk) in scanlines.chunks(65_535).enumerate() {
        let final_block = index == (scanlines.len() - 1) / 65_535;
        compressed.push(if final_block { 1 } else { 0 });
        let len = chunk.len() as u16;
        compressed.extend_from_slice(&len.to_le_bytes());
        compressed.extend_from_slice(&(!len).to_le_bytes());
        compressed.extend_from_slice(chunk);
    }
    compressed.extend_from_slice(&adler32(&scanlines).to_be_bytes());
    let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10];
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    append_chunk(&mut png, b"IHDR", &header);
    append_chunk(&mut png, b"IDAT", &compressed);
    append_chunk(&mut png, b"IEND", &[]);
    Ok(png)
}

fn append_chunk(output: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    output.extend_from_slice(&(data.len() as u32).to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    output.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}
fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for byte in bytes {
        a = (a + *byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
#[path = "../../tests/windows_active_window_screenshot_tests.rs"]
mod tests;
