//! Detects the CPU/RAM (and, where available, GPU) resources local
//! inference has to work with, so model/context sizing can be a decision
//! made from real numbers instead of a fixed guess baked into the binary.
//!
//! This is deliberately conservative about GPU: the bundled llama.cpp
//! engine is built CPU-only today (no `cuda`/`vulkan` backend), so
//! `gpu` is always `None` — reporting a GPU Cronicle cannot actually
//! offload to would be actively misleading. Wiring a GPU backend in is
//! future work (see `README.md`'s "Known limitations"); this module's
//! job is only to describe what's true right now.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareProfile {
    /// Logical CPU count (`std::thread::available_parallelism()`), i.e.
    /// including hyperthreads/SMT — the same number `inference_thread_count`
    /// in `local_model_provider` bases its thread count on.
    pub logical_cores: usize,
    /// Total installed physical RAM, in bytes.
    pub total_ram_bytes: u64,
    /// RAM currently free/available for new allocations, in bytes. This is
    /// a point-in-time snapshot — re-profile before each load decision
    /// rather than caching it, since it changes as other applications run.
    pub available_ram_bytes: u64,
    /// Always `None` today (see module docs) — reserved for when a GPU
    /// backend is actually wired in.
    pub gpu: Option<GpuInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuInfo {
    pub total_vram_bytes: u64,
    pub available_vram_bytes: u64,
}

impl HardwareProfile {
    /// Snapshots current hardware/memory state. Never fails: platform
    /// queries that are unavailable or error fall back to conservative
    /// defaults (see `platform::memory_status`) rather than propagating an
    /// error a caller would have to handle just to decide whether it's
    /// safe to load a model.
    pub fn detect() -> Self {
        let logical_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let (total_ram_bytes, available_ram_bytes) = platform::memory_status();
        Self {
            logical_cores,
            total_ram_bytes,
            available_ram_bytes,
            gpu: None,
        }
    }
}

#[cfg(windows)]
mod platform {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    /// `(total_ram_bytes, available_ram_bytes)` via `GlobalMemoryStatusEx`.
    /// Returns `(0, 0)` on API failure — treated by `MemoryPlanner` as "no
    /// information available", which biases toward the smaller/safer
    /// configuration rather than assuming memory is plentiful.
    pub fn memory_status() -> (u64, u64) {
        let mut status = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: `status` is zero-initialized with `dwLength` set as the
        // API requires; `GlobalMemoryStatusEx` only writes through the
        // pointer, matching the struct's declared size.
        match unsafe { GlobalMemoryStatusEx(&mut status) } {
            Ok(()) => (status.ullTotalPhys, status.ullAvailPhys),
            Err(_) => (0, 0),
        }
    }
}

#[cfg(not(windows))]
mod platform {
    pub fn memory_status() -> (u64, u64) {
        (0, 0)
    }
}

#[cfg(test)]
#[path = "tests/hardware_profiler_tests.rs"]
mod tests;
