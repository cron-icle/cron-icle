//! macOS-specific location for the pointer file that remembers the
//! user-chosen data directory. Reserved for a future macOS build, mirroring
//! `windows.rs`.

use std::path::{Path, PathBuf};

pub(super) fn pointer_file() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join("Library")
        .join("Application Support")
        .join("Chronicle")
        .join("data_dir.txt")
}

/// Reserved for a future macOS build (statvfs-based free-space check,
/// mirroring `windows.rs`'s `GetDiskFreeSpaceExW`). `None` here just skips
/// the pre-flight space check rather than blocking a move.
pub(super) fn available_space(_path: &Path) -> Option<u64> {
    None
}
