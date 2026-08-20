//! Windows-specific location for the pointer file that remembers the
//! user-chosen data directory. This is just a pointer — a few bytes of text
//! — not the data itself, so it lives in the fixed per-user `%APPDATA%`
//! location rather than needing a choice of its own.

use std::path::{Path, PathBuf};

pub(super) fn pointer_file() -> PathBuf {
    let base = std::env::var("APPDATA").unwrap_or_else(|_| ".".into());
    PathBuf::from(base).join("Chronicle").join("data_dir.txt")
}

/// Free space, in bytes, available to the current user on the volume
/// containing `path` — via `GetDiskFreeSpaceExW`, which (unlike total volume
/// capacity) accounts for per-user disk quotas.
pub(super) fn available_space(path: &Path) -> Option<u64> {
    use ::windows::core::HSTRING;
    use ::windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let wide = HSTRING::from(path.to_string_lossy().as_ref());
    let mut free_available_to_caller = 0u64;
    unsafe {
        GetDiskFreeSpaceExW(&wide, Some(&mut free_available_to_caller), None, None).ok()?;
    }
    Some(free_available_to_caller)
}
