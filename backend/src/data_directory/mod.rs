//! Where Chronicle stores its data — chosen explicitly by the user, from
//! Settings, not forced on app start.
//!
//! Everything Chronicle writes to disk (the sqlite event database, the
//! downloaded llama.cpp model files) lives under one directory instead of
//! being scattered into the install folder or a fixed path the user never
//! agreed to. There is deliberately no default: until the user picks one
//! (via the Settings panel, when they set up local AI), `current()` simply
//! returns `None` and Chronicle runs in a temporary, non-persistent mode —
//! it does not block startup on a folder-choose dialog. The choice, once
//! made, is remembered in a small pointer file whose OS location is
//! platform-specific (see `windows.rs`, `mac.rs`).
//!
//! Whatever folder the user picks, Chronicle never writes directly into it:
//! it creates (and, for both storage and retrieval, only ever operates on) a
//! `chronicle` subfolder underneath. The picked folder is often a general
//! one — a user's existing "Data" or "Documents" drive root, say — and
//! writing loose files straight into it, or later deleting siblings the
//! user didn't expect deleted, would be careless.

#[cfg(not(windows))]
mod mac;
#[cfg(windows)]
mod windows;

#[cfg(windows)]
use windows as platform;
#[cfg(not(windows))]
use mac as platform;

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const CHRONICLE_SUBFOLDER: &str = "chronicle";

fn read_pointer() -> Option<PathBuf> {
    let contents = std::fs::read_to_string(platform::pointer_file()).ok()?;
    let path = PathBuf::from(contents.trim());
    if path.as_os_str().is_empty() {
        None
    } else {
        Some(path)
    }
}

fn write_pointer(root: &Path) -> std::io::Result<()> {
    let pointer = platform::pointer_file();
    if let Some(parent) = pointer.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(pointer, root.to_string_lossy().as_bytes())
}

fn chronicle_subfolder(root: &Path) -> PathBuf {
    root.join(CHRONICLE_SUBFOLDER)
}

fn cell() -> &'static Mutex<Option<PathBuf>> {
    static CELL: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    CELL.get_or_init(|| {
        let configured = read_pointer()
            .filter(|root| chronicle_subfolder(root).is_dir())
            .map(|root| chronicle_subfolder(&root));
        Mutex::new(configured)
    })
}

/// The `chronicle` subfolder under the user-chosen root directory, if the
/// user has chosen one yet. `None` means Chronicle is running in a
/// temporary, non-persistent mode until Settings is used to pick a folder —
/// this never blocks or prompts on its own.
pub fn current() -> Option<PathBuf> {
    cell().lock().unwrap().clone()
}

/// Path to the sqlite event database, under the chosen data directory. Only
/// call once `current()` is known to be `Some` — every caller in this
/// codebase is gated on that (see `AppState::initialize`).
pub fn database_file() -> PathBuf {
    current()
        .expect("database_file() called before a data directory was chosen")
        .join("chronicle.db")
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Total size, in bytes, of every file under `path`.
fn directory_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| match entry.file_type() {
            Ok(file_type) if file_type.is_dir() => directory_size(&entry.path()),
            Ok(file_type) if file_type.is_file() => {
                entry.metadata().map(|metadata| metadata.len()).unwrap_or(0)
            }
            _ => 0,
        })
        .sum()
}

/// Recursively copies every entry under `src` into `dest` (which must
/// already exist), preserving relative structure, reporting cumulative
/// bytes copied so far against `total` after every file.
fn copy_dir_recursive(
    src: &Path,
    dest: &Path,
    copied: &mut u64,
    total: u64,
    on_progress: &mut dyn FnMut(u64, u64),
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = dest.join(entry.file_name());
        if file_type.is_dir() {
            std::fs::create_dir_all(&target)?;
            copy_dir_recursive(&entry.path(), &target, copied, total, on_progress)?;
        } else if file_type.is_file() {
            *copied += std::fs::copy(entry.path(), &target)?;
            on_progress(*copied, total);
        }
    }
    Ok(())
}

/// Sets or moves the data directory to `new_root` (a user-picked
/// destination — same "must be a real, explicitly chosen path, no default"
/// rule as `choose_new`), copying over whatever is already there if a data
/// directory was already configured.
///
/// When nothing was configured yet, this is exactly `choose_new` (no data to
/// move). Otherwise it checks free space at the destination before copying a
/// single byte, so a too-small target fails fast with a clear message
/// instead of partway through a multi-gigabyte copy; copies rather than
/// renames (a rename fails outright across drives — e.g. moving from `C:` to
/// a `D:` data disk); and only removes the old copy after every file has
/// landed safely in the new location. The caller is responsible for having
/// stopped anything that holds these files open (capture threads, the
/// llama.cpp servers, the database connection) before calling this —
/// copying files still being written by a live connection would race.
pub fn relocate_or_set(new_root: &Path, mut on_progress: impl FnMut(u64, u64)) -> Result<(), String> {
    if new_root.as_os_str().is_empty() {
        return Err("no destination directory was provided".into());
    }
    let Some(current) = current() else {
        let dir = chronicle_subfolder(new_root);
        std::fs::create_dir_all(&dir)
            .map_err(|error| format!("failed to create {}: {error}", dir.display()))?;
        write_pointer(new_root)
            .map_err(|error| format!("failed to remember the chosen data directory: {error}"))?;
        *cell().lock().unwrap() = Some(dir);
        return Ok(());
    };
    let dest = chronicle_subfolder(new_root);
    if dest == current {
        return Ok(());
    }
    std::fs::create_dir_all(&dest)
        .map_err(|error| format!("failed to create {}: {error}", dest.display()))?;

    let total = directory_size(&current);
    if let Some(available) = platform::available_space(new_root) {
        if available < total {
            return Err(format!(
                "the chosen folder doesn't have enough free space: needs {}, only {} available",
                format_bytes(total),
                format_bytes(available)
            ));
        }
    }

    copy_dir_recursive(&current, &dest, &mut 0, total, &mut on_progress)
        .map_err(|error| format!("failed to copy data to {}: {error}", dest.display()))?;
    write_pointer(new_root)
        .map_err(|error| format!("failed to remember the new data directory: {error}"))?;
    if let Err(error) = std::fs::remove_dir_all(&current) {
        tracing::warn!(%error, path = %current.display(), "moved data to the new directory but failed to remove the old copy; remove it manually if disk space matters");
    }
    *cell().lock().unwrap() = Some(dest);
    Ok(())
}
