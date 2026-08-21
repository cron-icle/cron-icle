//! Decides whether a model can be loaded safely, and at what context size,
//! from real available-memory numbers (`hardware_profiler`) instead of
//! always requesting a fixed context and hoping the machine survives.
//!
//! The estimate here is necessarily approximate: precise KV-cache sizing
//! needs a model's actual layer count / head dimensions, which are only
//! known *after* the GGUF is loaded (see `native_inference`'s
//! `GenerationEngine`) — by which point the weights are already resident
//! and it's too late to decide not to load them. `estimate_required_bytes`
//! instead scales a KV-cache-per-token estimate off the model *file size*,
//! which correlates with layer count × embedding width closely enough to
//! act as a load-time safety gate. It intentionally overestimates rather
//! than underestimates: this is a guard against OOM, not a precise budget.

use crate::hardware_profiler::HardwareProfile;

/// Fixed overhead for compute buffers, activation scratch space, and the
/// rest of Chronicle's own memory use (capture, SQLite, the Tauri/WebView2
/// process) that isn't the model itself. Conservative but not paranoid —
/// derived from the compute-buffer sizes logged during real GGUF loads in
/// this codebase's own testing (tens to a few hundred MiB depending on
/// context size), rounded up.
const FIXED_OVERHEAD_BYTES: u64 = 512 * 1024 * 1024;

/// Estimated KV-cache bytes per context token, per GB of model file size.
/// Calibrated loosely against real load logs from this codebase's testing
/// (a ~4B Q4 model's KV cache ran roughly 1 MiB per ~35 context tokens at
/// full context) — deliberately rounded toward "a bit too much" so this
/// stays a safety margin, not an exact accounting.
const KV_BYTES_PER_TOKEN_PER_GB: f64 = 40_000.0;

/// Context sizes tried in order, largest first, when the requested size
/// doesn't fit — mirrors `SERVER_CONTEXT_SIZE` in `local_model_provider`
/// (8192) as the top end, halving down to a floor still useful for a
/// single short event.
pub const CONTEXT_SIZE_LADDER: &[u32] = &[8192, 4096, 2048, 1024];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryPlan {
    pub context_size: u32,
    pub estimated_bytes: u64,
    pub available_bytes: u64,
}

/// Estimates total RAM a model load would need: file size (weights, mmap'd
/// roughly 1:1) plus a KV cache scaled by context size and model size, plus
/// fixed overhead for everything else running in the process.
pub fn estimate_required_bytes(model_file_bytes: u64, context_size: u32) -> u64 {
    let model_gb = model_file_bytes as f64 / 1_000_000_000.0;
    let kv_cache_bytes = (context_size as f64) * KV_BYTES_PER_TOKEN_PER_GB * model_gb;
    model_file_bytes + kv_cache_bytes.round() as u64 + FIXED_OVERHEAD_BYTES
}

/// Picks the largest context size from `CONTEXT_SIZE_LADDER` (capped at
/// `requested_context_size`) that fits within `profile`'s available RAM,
/// leaving `SAFETY_MARGIN` unused for the rest of the system. Returns
/// `None` if even the smallest rung doesn't fit — the caller's signal to
/// not load this model at all rather than load it and risk the OS running
/// out of memory.
pub fn plan_load(
    model_file_bytes: u64,
    requested_context_size: u32,
    profile: &HardwareProfile,
) -> Option<MemoryPlan> {
    /// Fraction of available RAM this planner will actually commit to a
    /// model load; the rest stays free for the OS and the rest of
    /// Chronicle. Not a hard OS limit — a deliberate margin against the
    /// estimate above being wrong in the unsafe direction.
    const SAFETY_MARGIN: f64 = 0.8;
    if profile.available_ram_bytes == 0 {
        // No usable reading from the platform (see `hardware_profiler`'s
        // `memory_status` failure case) — nothing safe can be concluded,
        // so don't claim a plan fits when we have no evidence either way.
        return None;
    }
    let usable_bytes = (profile.available_ram_bytes as f64 * SAFETY_MARGIN) as u64;
    CONTEXT_SIZE_LADDER
        .iter()
        .copied()
        .filter(|&size| size <= requested_context_size)
        .find_map(|context_size| {
            let estimated_bytes = estimate_required_bytes(model_file_bytes, context_size);
            (estimated_bytes <= usable_bytes).then_some(MemoryPlan {
                context_size,
                estimated_bytes,
                available_bytes: profile.available_ram_bytes,
            })
        })
}

/// Scales a worker's batch size down from `max_batch_size` when available
/// RAM is tight, instead of always claiming the same fixed batch
/// regardless of how much memory is actually free. Thresholds are in GiB
/// of *available* (not total) RAM, since that's what a batch's transient
/// allocations (numbered-prompt buffers, embedding sequences) compete for
/// against everything else running on the machine right now.
pub fn adaptive_batch_size(max_batch_size: usize, profile: &HardwareProfile) -> usize {
    const GIB: u64 = 1024 * 1024 * 1024;
    let available_gib = profile.available_ram_bytes / GIB;
    let scaled = if available_gib >= 8 {
        max_batch_size
    } else if available_gib >= 4 {
        (max_batch_size / 2).max(1)
    } else if available_gib >= 2 {
        1
    } else {
        // Unknown (0) or genuinely very low memory: don't refuse to batch
        // at all (a single item still has to be processed one way or
        // another), just don't compound it with siblings.
        1
    };
    scaled.clamp(1, max_batch_size.max(1))
}

/// Picks the smallest context size from `CONTEXT_SIZE_LADDER` that
/// comfortably holds `estimated_tokens` (prompt + response headroom),
/// falling back to the largest rung if nothing smaller fits. Chronicle's
/// prompts vary a lot in size — a single short window-focus event is a
/// fraction of an 8-batch numbered prompt — and requesting less context for
/// the small, common case means a smaller KV cache allocation and faster
/// context setup per request instead of always paying for the worst case.
pub fn context_size_for_tokens(estimated_tokens: u32) -> u32 {
    CONTEXT_SIZE_LADDER
        .iter()
        .copied()
        .rev() // ascending: try the smallest rung first
        .find(|&size| size >= estimated_tokens)
        .unwrap_or_else(|| CONTEXT_SIZE_LADDER.first().copied().unwrap_or(estimated_tokens))
}

#[cfg(test)]
#[path = "tests/memory_planner_tests.rs"]
mod tests;
