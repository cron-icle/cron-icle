//! In-process local inference via `llama-cpp-rs` bindings to llama.cpp,
//! replacing the HTTP client to a separately spawned `llama-server.exe`.
//!
//! This is not about raw speed — the actual matrix math still runs inside
//! the same llama.cpp C++ backend either way. What direct bindings buy is
//! control over the model/context lifecycle that an HTTP server hides
//! behind a black box: exactly when weights are loaded into memory, exactly
//! what context size and thread count a request uses, and the ability to
//! free that memory deterministically instead of a server process holding
//! it for as long as it happens to stay running. See `README.md`'s "Local
//! AI engine" section for how this fits into the rest of the pipeline.
//!
//! `LlamaBackend::init()` is process-global and expensive to repeat, so it's
//! initialized once behind `backend()`. Model weights (`LlamaModel`) are the
//! expensive-to-load, cheap-to-reuse part and are kept in `GenerationEngine`
//! / `EmbeddingEngine`; the `LlamaContext` (KV cache buffers) is cheap to
//! allocate and is created fresh per request rather than reused, since every
//! Chronicle inference call is an independent single-turn request with no
//! conversation state to preserve across calls.

use encoding_rs::UTF_8;
use llama_cpp_rs::context::params::LlamaContextParams;
use llama_cpp_rs::llama_backend::LlamaBackend;
use llama_cpp_rs::llama_batch::LlamaBatch;
use llama_cpp_rs::model::params::LlamaModelParams;
use llama_cpp_rs::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_rs::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_rs::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, OnceLock};

fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("llama.cpp backend failed to initialize"))
}

/// Resident text/vision-capable model plus the settings every request built
/// from it should use. Cheap to clone (an `Arc` around the loaded weights),
/// so a `ModelManager` (V3) can hand out the current generation
/// (`GenerationEngine`) or embedding (`EmbeddingEngine`) engine without
/// re-reading the GGUF file from disk.
#[derive(Clone)]
pub struct GenerationEngine {
    model: Arc<LlamaModel>,
    chat_template: LlamaChatTemplate,
    n_ctx: NonZeroU32,
    n_threads: i32,
}

#[derive(Clone)]
pub struct EmbeddingEngine {
    model: Arc<LlamaModel>,
    n_ctx: NonZeroU32,
    n_threads: i32,
}

/// Sampling temperature Chronicle has always used for structured-JSON
/// extraction: low enough to stay close to the model's most likely reading
/// of the event, not zero (pure greedy can get stuck repeating itself on
/// small quantized models more than a slightly-random low-temperature
/// sample does).
const SAMPLING_TEMPERATURE: f32 = 0.2;

// A JSON grammar sampler (`LlamaSampler::grammar`) was tried here to
// constrain decoding to valid JSON — matching what `llama-server`'s
// `response_format: {"type":"json_object"}` gave the HTTP path for free —
// but a hand-written GBNF grammar triggered a native `GGML_ASSERT
// (!stacks.empty())` abort in this llama-cpp-2/llama.cpp version's grammar
// engine (`llama-grammar.cpp`), which crashes the whole process rather than
// failing gracefully. That's a strictly worse failure mode than the
// problem it was meant to solve, so it's reverted for now. Native JSON
// output currently relies only on prompt instructions + `max_tokens` +
// `parse_and_validate_model_json`/`parse_batch_response` catching malformed
// output and letting the existing queue retry handle it — same reliability
// envelope llama-server callers had before `response_format` existed.
// Revisit with a grammar built via llama.cpp's own `json_schema_to_grammar`
// (schema-driven, less error-prone than hand-written GBNF) rather than a
// hand-rolled grammar string.

fn load_model(model_path: &Path, n_gpu_layers: u32) -> Result<Arc<LlamaModel>, String> {
    let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
    let model = LlamaModel::load_from_file(backend(), model_path, &params)
        .map_err(|error| format!("failed to load model {}: {error}", model_path.display()))?;
    Ok(Arc::new(model))
}

impl GenerationEngine {
    pub fn load(
        model_path: &Path,
        n_gpu_layers: u32,
        n_ctx: NonZeroU32,
        n_threads: i32,
    ) -> Result<Self, String> {
        let model = load_model(model_path, n_gpu_layers)?;
        let chat_template = model
            .chat_template(None)
            .map_err(|error| format!("model has no usable chat template: {error}"))?;
        Ok(Self { model, chat_template, n_ctx, n_threads })
    }

    /// The context size this engine was actually loaded with — used by the
    /// `generation_engine` cache to decide whether a resident engine still
    /// has enough headroom for a new request or needs reloading with more.
    pub fn n_ctx(&self) -> NonZeroU32 {
        self.n_ctx
    }

    fn new_context(&self) -> Result<llama_cpp_rs::context::LlamaContext<'_>, String> {
        let params = LlamaContextParams::default()
            .with_n_ctx(Some(self.n_ctx))
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads);
        self.model
            .new_context(backend(), params)
            .map_err(|error| format!("failed to create inference context: {error}"))
    }

    /// Renders `prompt` as a single user turn through the model's own chat
    /// template (so Gemma 3's instruction formatting is applied the same
    /// way `--jinja` did for the HTTP server), then greedily-with-low-
    /// temperature decodes up to `max_tokens` tokens or until the model
    /// emits an end-of-generation token, whichever comes first — the same
    /// `max_tokens` bound `local_model_provider`'s HTTP client enforces, for
    /// the same reason: one non-terminating generation must not be able to
    /// pin a worker thread indefinitely.
    pub fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String, String> {
        let message = LlamaChatMessage::new("user".to_string(), prompt.to_string())
            .map_err(|error| format!("invalid prompt for chat message: {error}"))?;
        let rendered = self
            .model
            .apply_chat_template(&self.chat_template, &[message], true)
            .map_err(|error| format!("failed to apply chat template: {error}"))?;

        let mut ctx = self.new_context()?;
        let tokens = self
            .model
            .str_to_token(&rendered, AddBos::Always)
            .map_err(|error| format!("failed to tokenize prompt: {error}"))?;
        if tokens.len() as u32 >= self.n_ctx.get() {
            return Err(format!(
                "prompt ({} tokens) does not fit in context ({} tokens)",
                tokens.len(),
                self.n_ctx.get()
            ));
        }

        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        let last_index = tokens.len() as i32 - 1;
        for (i, token) in tokens.into_iter().enumerate() {
            batch
                .add(token, i as i32, &[0], i as i32 == last_index)
                .map_err(|error| format!("failed to build prompt batch: {error}"))?;
        }
        ctx.decode(&mut batch)
            .map_err(|error| format!("failed to evaluate prompt: {error}"))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(SAMPLING_TEMPERATURE),
            LlamaSampler::dist(rand_seed()),
        ]);
        let mut decoder = UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();
        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|error| format!("failed to decode generated token: {error}"))?;
            output.push_str(&piece);

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|error| format!("failed to extend generation batch: {error}"))?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|error| format!("failed to evaluate generated token: {error}"))?;
        }
        Ok(output)
    }
}

impl EmbeddingEngine {
    pub fn load(model_path: &Path, n_gpu_layers: u32, n_ctx: NonZeroU32, n_threads: i32) -> Result<Self, String> {
        let model = load_model(model_path, n_gpu_layers)?;
        Ok(Self { model, n_ctx, n_threads })
    }

    /// Embeds every input in one context, each as its own sequence, then
    /// reads back one pooled vector per sequence — the native equivalent of
    /// the HTTP client's single `/v1/embeddings` call with an array `input`.
    /// Pooling type is left at the model's own GGUF-declared default
    /// (`Unspecified`) rather than forced, matching what `llama-server`
    /// does when the caller doesn't override it.
    pub fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let params = LlamaContextParams::default()
            .with_n_ctx(Some(self.n_ctx))
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads)
            .with_n_seq_max(inputs.len() as u32)
            .with_embeddings(true);
        let mut ctx = self
            .model
            .new_context(backend(), params)
            .map_err(|error| format!("failed to create embedding context: {error}"))?;

        let tokenized: Vec<Vec<llama_cpp_rs::token::LlamaToken>> = inputs
            .iter()
            .map(|input| {
                self.model
                    .str_to_token(input, AddBos::Always)
                    .map_err(|error| format!("failed to tokenize embedding input: {error}"))
            })
            .collect::<Result<_, _>>()?;

        let total_tokens: usize = tokenized.iter().map(Vec::len).sum();
        if total_tokens as u32 >= self.n_ctx.get() {
            return Err(format!(
                "embedding batch ({total_tokens} tokens across {} inputs) does not fit in context ({} tokens)",
                inputs.len(),
                self.n_ctx.get()
            ));
        }

        let mut batch = LlamaBatch::new(total_tokens.max(512), inputs.len() as i32);
        for (seq_id, tokens) in tokenized.iter().enumerate() {
            let last_index = tokens.len() as i32 - 1;
            for (i, token) in tokens.iter().enumerate() {
                batch
                    .add(*token, i as i32, &[seq_id as i32], i as i32 == last_index)
                    .map_err(|error| format!("failed to build embedding batch: {error}"))?;
            }
        }
        ctx.decode(&mut batch)
            .map_err(|error| format!("failed to evaluate embedding batch: {error}"))?;

        (0..inputs.len())
            .map(|seq_id| {
                ctx.embeddings_seq_ith(seq_id as i32)
                    .map(<[f32]>::to_vec)
                    .map_err(|error| format!("failed to read embedding {seq_id}: {error}"))
            })
            .collect()
    }
}

/// Resident vision-capable engine: the same Gemma 3 chat model as
/// `GenerationEngine` plus an `MtmdContext` loaded from the mmproj file,
/// so a screenshot can be embedded and decoded in the same process instead
/// of being base64-posted to a separately spawned `llama-server --mmproj`
/// (see `local_model_provider::analyze_image`). `MtmdContext` only stores a
/// raw pointer with no lifetime tied to the `LlamaModel` it was built from,
/// so it's paired with the `Arc<LlamaModel>` it depends on here to keep the
/// model alive for at least as long as the context does.
#[derive(Clone)]
pub struct VisionEngine {
    model: Arc<LlamaModel>,
    chat_template: LlamaChatTemplate,
    mtmd: Arc<MtmdContext>,
    n_ctx: NonZeroU32,
    n_threads: i32,
}

impl VisionEngine {
    pub fn load(
        model_path: &Path,
        mmproj_path: &Path,
        n_gpu_layers: u32,
        n_ctx: NonZeroU32,
        n_threads: i32,
    ) -> Result<Self, String> {
        let model = load_model(model_path, n_gpu_layers)?;
        let chat_template = model
            .chat_template(None)
            .map_err(|error| format!("model has no usable chat template: {error}"))?;
        let mtmd_params = MtmdContextParams {
            use_gpu: n_gpu_layers > 0,
            print_timings: false,
            n_threads,
            media_marker: std::ffi::CString::new(llama_cpp_rs::mtmd::mtmd_default_marker())
                .expect("default media marker has no interior NUL"),
            image_min_tokens: -1,
            image_max_tokens: -1,
        };
        let mtmd = MtmdContext::init_from_file(&mmproj_path.to_string_lossy(), &model, &mtmd_params)
            .map_err(|error| format!("failed to load mmproj {}: {error}", mmproj_path.display()))?;
        Ok(Self { model, chat_template, mtmd: Arc::new(mtmd), n_ctx, n_threads })
    }

    pub fn n_ctx(&self) -> NonZeroU32 {
        self.n_ctx
    }

    /// Renders `prompt` through the chat template with the image's media
    /// marker appended (if the caller didn't already place one), tokenizes
    /// text+image together via `MtmdContext::tokenize`, prefills both
    /// through `MtmdInputChunks::eval_chunks` (which internally runs
    /// `mtmd_encode` for the image chunk and `llama_decode` for the rest),
    /// then decodes up to `max_tokens` the same way `GenerationEngine::generate`
    /// does for text-only prompts.
    pub fn generate_with_image(&self, image_bytes: &[u8], prompt: &str, max_tokens: u32) -> Result<String, String> {
        let marker = llama_cpp_rs::mtmd::mtmd_default_marker();
        let prompt_with_marker =
            if prompt.contains(marker) { prompt.to_string() } else { format!("{prompt}{marker}") };
        let message = LlamaChatMessage::new("user".to_string(), prompt_with_marker)
            .map_err(|error| format!("invalid prompt for chat message: {error}"))?;
        let rendered = self
            .model
            .apply_chat_template(&self.chat_template, &[message], true)
            .map_err(|error| format!("failed to apply chat template: {error}"))?;

        let bitmap = MtmdBitmap::from_buffer(&self.mtmd, image_bytes, false)
            .map_err(|error| format!("failed to decode image: {error}"))?;
        let input_text = MtmdInputText { text: rendered, add_special: true, parse_special: true };
        let chunks = self
            .mtmd
            .tokenize(input_text, &[&bitmap])
            .map_err(|error| format!("failed to tokenize image prompt: {error}"))?;

        let params = LlamaContextParams::default()
            .with_n_ctx(Some(self.n_ctx))
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads);
        let mut ctx = self
            .model
            .new_context(backend(), params)
            .map_err(|error| format!("failed to create inference context: {error}"))?;

        let mut n_past = chunks
            .eval_chunks(&self.mtmd, &ctx, 0, 0, self.n_ctx.get() as i32, true)
            .map_err(|error| format!("failed to evaluate image prompt: {error}"))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(SAMPLING_TEMPERATURE),
            LlamaSampler::dist(rand_seed()),
        ]);
        let mut decoder = UTF_8.new_decoder();
        let mut output = String::new();
        let mut batch = LlamaBatch::new(512, 1);
        // The prefill decode happened inside `eval_chunks`, opaque to this
        // function, so there's no local `LlamaBatch` whose last-token index
        // to sample from; -1 is llama.cpp's own convention for "logits of
        // the most recently decoded token", which is what the upstream mtmd
        // CLI example uses for exactly this first-token-after-prefill case.
        let mut next_index: i32 = -1;
        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, next_index);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|error| format!("failed to decode generated token: {error}"))?;
            output.push_str(&piece);

            batch.clear();
            batch
                .add(token, n_past, &[0], true)
                .map_err(|error| format!("failed to extend generation batch: {error}"))?;
            n_past += 1;
            ctx.decode(&mut batch)
                .map_err(|error| format!("failed to evaluate generated token: {error}"))?;
            next_index = batch.n_tokens() - 1;
        }
        Ok(output)
    }
}

/// Process-wide resident engines, lazily loaded on first use. Keyed by
/// model path so switching the configured model file reloads rather than
/// silently keeps serving the old weights.
///
/// The two engines have deliberately different lifecycle policies (see
/// module docs and `README.md`'s "Local AI engine" section): embeddings are
/// small, cheap to keep resident, and called on every processed event
/// (including once more right after every text-analysis call, since
/// `persist_semantic_result` always enqueues a follow-up embedding task —
/// see `asynchronous_processing_queue.rs`), so `EMBEDDING_ENGINE` never
/// idle-unloads. The generation model is the large one (a full Gemma 3 4B
/// is gigabytes resident) and sits unused for long stretches between
/// capture events, so `GENERATION_ENGINE` unloads after
/// `GENERATION_KEEP_ALIVE` of inactivity — freeing that memory back to the
/// rest of the system is the single biggest lever this whole native-
/// inference migration has over a permanently-running `llama-server`.
static GENERATION_ENGINE: OnceLock<std::sync::Mutex<Option<ResidentEngine<GenerationEngine>>>> = OnceLock::new();
static EMBEDDING_ENGINE: OnceLock<std::sync::Mutex<Option<(std::path::PathBuf, EmbeddingEngine)>>> =
    OnceLock::new();
/// Keyed by `(chat_model_path, mmproj_path)` since either file changing
/// invalidates the resident `MtmdContext`. Shares `GENERATION_KEEP_ALIVE`'s
/// unload policy via `sweep_idle_engines` for the same reason
/// `GENERATION_ENGINE` does: the vision model is the same multi-gigabyte
/// Gemma 3 weights, just with the projector attached, so it shouldn't sit
/// resident indefinitely either.
static VISION_ENGINE: OnceLock<std::sync::Mutex<Option<ResidentEngine<VisionEngine>>>> = OnceLock::new();

/// How long the generation engine stays loaded after its last use before
/// `sweep_idle_engines` unloads it. Deliberately longer than a single
/// worker batch-processing pause (`MIN_BATCH_PACING` in
/// `asynchronous_processing_queue.rs` is 300ms) so back-to-back batches
/// don't thrash reload; short enough that a genuinely idle machine gets the
/// memory back within a couple of worker poll cycles of the queue going
/// quiet.
const GENERATION_KEEP_ALIVE: std::time::Duration = std::time::Duration::from_secs(60);

struct ResidentEngine<T> {
    model_path: std::path::PathBuf,
    engine: T,
    last_used: std::time::Instant,
}

/// Coarse lifecycle state for a resident engine, exposed for telemetry
/// (V4) and diagnostics — not an internal control mechanism itself (the
/// actual load/unload decisions live in `generation_engine`/
/// `sweep_idle_engines`), just a read of where things currently stand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    Unloaded,
    /// Loaded and used within the keep-alive window.
    Ready,
    /// Loaded but past its keep-alive window — will be freed by the next
    /// `sweep_idle_engines` call rather than serving further requests from
    /// stale state.
    Idle,
}

pub fn generation_engine_state() -> ModelState {
    let Some(cell) = GENERATION_ENGINE.get() else {
        return ModelState::Unloaded;
    };
    match cell.lock().ok().and_then(|guard| guard.as_ref().map(|r| r.last_used)) {
        None => ModelState::Unloaded,
        Some(last_used) if last_used.elapsed() >= GENERATION_KEEP_ALIVE => ModelState::Idle,
        Some(_) => ModelState::Ready,
    }
}

/// Frees the generation engine's memory if it's gone unused past
/// `GENERATION_KEEP_ALIVE`. Cheap to call when there's nothing to unload
/// (a `Mutex` lock plus an `Instant` comparison), so
/// `run_processing_worker`'s idle poll loop calls it on every empty-queue
/// tick rather than needing a separate timer thread — the worker is
/// already the thing best positioned to know "no AI work has happened
/// recently."
pub fn sweep_idle_engines() {
    if let Some(cell) = GENERATION_ENGINE.get() {
        if let Ok(mut guard) = cell.lock() {
            if guard.as_ref().is_some_and(|r| r.last_used.elapsed() >= GENERATION_KEEP_ALIVE) {
                *guard = None; // dropping the Arc<LlamaModel> here releases the weights
            }
        }
    }
    if let Some(cell) = VISION_ENGINE.get() {
        if let Ok(mut guard) = cell.lock() {
            if guard.as_ref().is_some_and(|r| r.last_used.elapsed() >= GENERATION_KEEP_ALIVE) {
                *guard = None;
            }
        }
    }
}

/// Test-only: back-dates the resident generation engine's `last_used` so
/// `sweep_idle_engines`'s real unload path can be exercised deterministically
/// instead of actually sleeping past `GENERATION_KEEP_ALIVE` (60s) in a test.
#[cfg(test)]
fn force_generation_engine_stale_for_test() {
    if let Some(cell) = GENERATION_ENGINE.get() {
        if let Ok(mut guard) = cell.lock() {
            if let Some(resident) = guard.as_mut() {
                resident.last_used = std::time::Instant::now() - GENERATION_KEEP_ALIVE - std::time::Duration::from_secs(1);
            }
        }
    }
}

/// Resolves the context size an engine should actually load with:
/// `requested_n_ctx` if the `memory_planner` finds it (or something smaller
/// from `CONTEXT_SIZE_LADDER`) safe against currently available RAM, an
/// error otherwise. This is where V2's hardware/memory awareness actually
/// changes behavior rather than just being available to call — every load
/// through `generation_engine`/`embedding_engine` goes through it.
fn safe_context_size(model_path: &Path, requested_n_ctx: NonZeroU32) -> Result<NonZeroU32, String> {
    let model_bytes = std::fs::metadata(model_path)
        .map_err(|error| format!("failed to read model file size for {}: {error}", model_path.display()))?
        .len();
    let profile = crate::hardware_profiler::HardwareProfile::detect();
    let plan = crate::memory_planner::plan_load(model_bytes, requested_n_ctx.get(), &profile).ok_or_else(|| {
        format!(
            "not enough available memory to safely load {} ({} MB) with any context size down to {}: {} MB available",
            model_path.display(),
            model_bytes / (1024 * 1024),
            crate::memory_planner::CONTEXT_SIZE_LADDER.last().copied().unwrap_or(0),
            profile.available_ram_bytes / (1024 * 1024)
        )
    })?;
    Ok(NonZeroU32::new(plan.context_size).expect("plan_load only returns nonzero context sizes"))
}

pub fn generation_engine(model_path: &Path, n_ctx: NonZeroU32, n_threads: i32) -> Result<GenerationEngine, String> {
    let cell = GENERATION_ENGINE.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = cell.lock().map_err(|_| "generation engine lock poisoned".to_string())?;
    // Adaptive context sizing (see `local_model_provider::generate_chat`)
    // means different calls can request different context sizes for the
    // same model. Reusing a resident engine that was loaded with *less*
    // context than this call needs would silently truncate what the
    // request can hold, so only reuse when the resident engine has at
    // least as much headroom as requested — otherwise reload with the
    // larger size (which also re-runs the memory-safety check below for
    // the new size, not the old one).
    if let Some(resident) = guard.as_mut() {
        if resident.model_path == model_path && resident.engine.n_ctx() >= n_ctx {
            resident.last_used = std::time::Instant::now();
            return Ok(resident.engine.clone());
        }
    }
    let safe_n_ctx = safe_context_size(model_path, n_ctx)?;
    let engine = GenerationEngine::load(model_path, 0, safe_n_ctx, n_threads)?;
    *guard = Some(ResidentEngine {
        model_path: model_path.to_path_buf(),
        engine: engine.clone(),
        last_used: std::time::Instant::now(),
    });
    Ok(engine)
}

pub fn vision_engine(
    model_path: &Path,
    mmproj_path: &Path,
    n_ctx: NonZeroU32,
    n_threads: i32,
) -> Result<VisionEngine, String> {
    let cell = VISION_ENGINE.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = cell.lock().map_err(|_| "vision engine lock poisoned".to_string())?;
    if let Some(resident) = guard.as_mut() {
        if resident.model_path == model_path && resident.engine.n_ctx() >= n_ctx {
            resident.last_used = std::time::Instant::now();
            return Ok(resident.engine.clone());
        }
    }
    let safe_n_ctx = safe_context_size(model_path, n_ctx)?;
    let engine = VisionEngine::load(model_path, mmproj_path, 0, safe_n_ctx, n_threads)?;
    *guard = Some(ResidentEngine {
        model_path: model_path.to_path_buf(),
        engine: engine.clone(),
        last_used: std::time::Instant::now(),
    });
    Ok(engine)
}

pub fn embedding_engine(model_path: &Path, n_ctx: NonZeroU32, n_threads: i32) -> Result<EmbeddingEngine, String> {
    let cell = EMBEDDING_ENGINE.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = cell.lock().map_err(|_| "embedding engine lock poisoned".to_string())?;
    if let Some((path, engine)) = guard.as_ref() {
        if path == model_path {
            return Ok(engine.clone());
        }
    }
    let n_ctx = safe_context_size(model_path, n_ctx)?;
    let engine = EmbeddingEngine::load(model_path, 0, n_ctx, n_threads)?;
    *guard = Some((model_path.to_path_buf(), engine.clone()));
    Ok(engine)
}

fn rand_seed() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(1234)
}

#[cfg(test)]
#[path = "tests/native_inference_tests.rs"]
mod tests;
