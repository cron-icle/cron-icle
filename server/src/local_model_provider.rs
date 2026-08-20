//! Local inference over Gemma 3 (chat/vision) and EmbeddingGemma, both run
//! in-process via `native_inference` (Chronicle's own fork of
//! `llama-cpp-rs`, at `E:\llama-cpp-rs`, with the `mtmd` multimodal feature
//! enabled) rather than through a separately spawned `llama-server` HTTP
//! server. The GGUF model files (and mmproj projector) live under
//! `<data dir>\llama\models` (see `engine_paths`), where `<data dir>` is the
//! folder the user chose on first run (see `data_directory`), and are
//! downloaded once by `local_inference_setup`; nothing here downloads
//! anything itself.
//!
//! `llama-server.exe` is still bundled and spawned by
//! `start_chat_server_if_needed`/`start_embed_server_if_needed` for now, but
//! nothing in this provider talks to it over HTTP any more — see the
//! `native_inference` module for the actual inference path. Retiring that
//! spawn (and the bundled binary/download) is tracked separately.

use crate::embedding_provider::TextEmbedder;
use crate::local_semantic_processing::{
    parse_and_validate_model_json, validate_image_input, LocalSemanticAnalyzer, SemanticModelOutput,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

/// Context window size (in tokens) both bundled servers are started with.
/// `analyze_text_batch` concatenates up to `MAX_MODEL_BATCH_SIZE` event
/// contexts plus `analyze_image`'s base64-embedded screenshots into a single
/// prompt; llama.cpp's own default context (often 4096 or the model's
/// training context) is not reliably large enough for that, and a prompt
/// that overflows it is rejected outright rather than truncated.
const SERVER_CONTEXT_SIZE: u32 = 8192;

/// Upper bound on generated tokens per chat/vision request. Without this,
/// `n_predict` defaults to `-1` (generate until end-of-sequence or the
/// context fills), so a single request the model can't naturally terminate
/// — a common failure mode for small quantized models asked for strict JSON
/// — pins one of the server's slots and the calling worker thread for as
/// long as the whole remaining context takes to fill. Capping it bounds
/// worst-case per-task latency, which is what keeps the queue moving under
/// load rather than stalling behind one bad generation. The structured
/// output this provider asks for (category/summary/entities/relationships/
/// confidence, optionally for up to `MAX_MODEL_BATCH_SIZE` items) fits well
/// inside this budget.
const MAX_RESPONSE_TOKENS: u32 = 1024;

/// Opens (creating/truncating) a log file under `<data dir>\llama\logs` for a
/// spawned server's stdout/stderr. Both streams were previously discarded
/// (`Stdio::null()`), which made every startup and inference failure from
/// `llama-server.exe` invisible — the process stays up and its port stays
/// reachable even when, for example, it can't apply the model's chat
/// template, so `chat_reachable()` reports healthy while every real request
/// fails. Logging to a file makes that diagnosable without changing the
/// "pending, not failed" behavior when the engine isn't installed yet.
/// Returns one `Stdio` per stream (stdout, stderr), both appending to the
/// same log file, so interleaved output stays in one place per server.
fn open_server_log(name: &str) -> (Stdio, Stdio) {
    let Some(path) = server_log_path(name) else {
        return (Stdio::null(), Stdio::null());
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match File::create(&path).and_then(|file| Ok((file.try_clone()?, file))) {
        Ok((out, err)) => (Stdio::from(out), Stdio::from(err)),
        Err(_) => (Stdio::null(), Stdio::null()),
    }
}

/// Builds the argument list for the chat/vision `llama-server`, factored out
/// so the exact flags can be asserted on in tests without spawning a real
/// process. `--jinja` enables llama.cpp's Jinja chat-template engine, which
/// Gemma 3's chat template requires — without it, `/v1/chat/completions`
/// fails (empty/unsupported template) even though the server process stays
/// up and the port stays reachable, which is what made this failure
/// invisible before. `-c` raises the context window past llama.cpp's
/// default so a batched multi-event prompt (see `analyze_text_batch`) or an
/// embedded screenshot doesn't get rejected for overflowing it.
fn chat_server_args(chat_model: &Path, mmproj: &Path, host: &str, port: u16) -> Vec<String> {
    vec![
        "-m".into(),
        chat_model.to_string_lossy().into_owned(),
        "--mmproj".into(),
        mmproj.to_string_lossy().into_owned(),
        "--host".into(),
        host.into(),
        "--port".into(),
        port.to_string(),
        "--jinja".into(),
        "-c".into(),
        SERVER_CONTEXT_SIZE.to_string(),
        "-t".into(),
        inference_thread_count().to_string(),
    ]
}

/// Threads llama.cpp is told to use for generation. llama.cpp's own default
/// (`-1`) already resolves to the host's core count, but pinning it
/// explicitly makes the number visible/tunable here instead of buried in
/// the engine's own heuristics, and avoids over-subscribing on hybrid
/// (performance + efficiency core) CPUs where llama.cpp's auto-detection is
/// not always the count you'd actually pick. One core is held back for the
/// rest of Chronicle (capture hooks, the Tauri UI thread, SQLite) so local
/// inference never fully starves the app it's running inside.
fn inference_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(4)
}

/// Builds the argument list for the embedding `llama-server`. See
/// `chat_server_args` for why `--jinja` and `-c` are present; `--jinja` is
/// harmless for a pure-embedding model but keeps the two spawn paths
/// consistent, and `-c` matters here too since `embed_batch` sends one
/// request per batch of inputs.
fn embed_server_args(embed_model: &Path, host: &str, port: u16) -> Vec<String> {
    vec![
        "-m".into(),
        embed_model.to_string_lossy().into_owned(),
        "--host".into(),
        host.into(),
        "--port".into(),
        port.to_string(),
        "--embeddings".into(),
        "-c".into(),
        SERVER_CONTEXT_SIZE.to_string(),
        "-t".into(),
        inference_thread_count().to_string(),
    ]
}

fn server_log_path(name: &str) -> Option<PathBuf> {
    Some(
        crate::data_directory::current()?
            .join("llama")
            .join("logs")
            .join(format!("{name}.log")),
    )
}

/// Where the bundled engine (binary + models) lives and what its pieces are
/// named. A single source of truth shared by the provider (to run
/// inference) and `local_inference_setup` (to download/remove these same files).
pub mod engine_paths {
    use std::path::PathBuf;

    /// Display name for the chat/vision model file — also its filename.
    ///
    /// Sourced from `bartowski`'s GGUF re-upload rather than Google's
    /// official `google/gemma-3-4b-it-qat-q4_0-gguf` repo: Google's repo is
    /// access-gated (requires a Hugging Face login and accepting a license
    /// agreement), which returns HTTP 401 for the anonymous download this
    /// setup flow does. `bartowski`'s re-upload of the same weights is
    /// openly downloadable and is the community-standard mirror llama.cpp
    /// users are pointed to for exactly this reason.
    pub const CHAT_MODEL_FILE: &str = "google_gemma-3-4b-it-Q4_K_M.gguf";
    /// Multimodal projector required alongside the chat model for vision input.
    pub const MMPROJ_FILE: &str = "mmproj-google_gemma-3-4b-it-f16.gguf";
    /// Display name for the embedding model file — also its filename.
    pub const EMBED_MODEL_FILE: &str = "embeddinggemma-300M-Q8_0.gguf";

    pub const CHAT_MODEL_URL: &str = "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf";
    pub const MMPROJ_URL: &str = "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/mmproj-google_gemma-3-4b-it-f16.gguf";
    pub const EMBED_MODEL_URL: &str = "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf";

    /// `None` until the user has chosen a data directory from Settings —
    /// there is nowhere to put model files yet.
    fn base_dir() -> Option<PathBuf> {
        Some(crate::data_directory::current()?.join("llama"))
    }

    /// Where the `llama-server` binary and its DLLs live. Unlike the model
    /// weights below, the engine ships alongside the daemon binary itself
    /// (see `src-tauri/resources/llama/`) rather than downloaded at runtime,
    /// so this looks next to the running executable instead of under the
    /// data directory. Falls back to the source tree's `resources/llama`
    /// when running via `cargo run`, which doesn't copy resources next to
    /// the dev binary.
    pub fn runtime_dir() -> PathBuf {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                for candidate in [exe_dir.join("llama"), exe_dir.join("resources").join("llama")] {
                    if candidate.join("llama-server.exe").is_file() {
                        return candidate;
                    }
                }
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("llama")
    }
    pub fn models_dir() -> Option<PathBuf> {
        Some(base_dir()?.join("models"))
    }
    pub fn server_binary() -> PathBuf {
        runtime_dir().join("llama-server.exe")
    }
    pub fn chat_model() -> Option<PathBuf> {
        Some(models_dir()?.join(CHAT_MODEL_FILE))
    }
    pub fn mmproj() -> Option<PathBuf> {
        Some(models_dir()?.join(MMPROJ_FILE))
    }
    pub fn embed_model() -> Option<PathBuf> {
        Some(models_dir()?.join(EMBED_MODEL_FILE))
    }
    pub fn runtime_installed() -> bool {
        server_binary().is_file()
    }
    pub fn chat_model_installed() -> bool {
        chat_model().is_some_and(|path| path.is_file()) && mmproj().is_some_and(|path| path.is_file())
    }
    pub fn embed_model_installed() -> bool {
        embed_model().is_some_and(|path| path.is_file())
    }
}

/// One keep-alive `ureq` agent shared by every provider instance and every
/// worker thread. Reusing pooled connections instead of opening a fresh TCP
/// connection per inference call removes a full connect + slow-start round
/// trip from every request, and `ureq` correctly handles chunked transfer
/// encoding and HTTP status codes instead of guessing from a raw byte split.
pub(crate) fn shared_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(2)))
            .timeout_recv_response(Some(Duration::from_secs(120)))
            .build()
            .into()
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelStatus {
    pub chat_endpoint: String,
    pub embedding_endpoint: String,
    pub chat_model: String,
    pub embedding_model: String,
    pub chat_available: bool,
    pub embedding_available: bool,
}

#[derive(Debug, Clone)]
pub struct LlamaCppProvider {
    pub host: String,
    pub chat_port: u16,
    pub embed_port: u16,
    pub chat_model: String,
    pub embedding_model: String,
}

impl Default for LlamaCppProvider {
    fn default() -> Self {
        Self {
            host: std::env::var("CHRONICLE_LLAMA_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
            chat_port: std::env::var("CHRONICLE_LLAMA_CHAT_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8090),
            embed_port: std::env::var("CHRONICLE_LLAMA_EMBED_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8091),
            chat_model: engine_paths::CHAT_MODEL_FILE.to_string(),
            embedding_model: engine_paths::EMBED_MODEL_FILE.to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct BatchSemanticResponse {
    results: Vec<BatchSemanticItem>,
}
#[derive(Debug, Deserialize)]
struct BatchSemanticItem {
    index: usize,
    category: String,
    summary: String,
    entities: Vec<String>,
    relationships: Vec<String>,
    confidence: f32,
}

/// Parses and re-orders a numbered-prompt batch response (see
/// `LlamaCppProvider::analyze_text_batch`) into per-input results, indexed
/// by the model's own reported `index` rather than response order — pulled
/// out as a pure function, independent of the transport that produced
/// `content` (HTTP or, now, the native in-process engine), so this parsing
/// contract can be tested without spinning up either.
fn parse_batch_response(content: &str, expected_len: usize) -> Result<Vec<SemanticModelOutput>, String> {
    let response: BatchSemanticResponse =
        serde_json::from_str(content).map_err(|e| format!("invalid batch semantic JSON: {e}"))?;
    if response.results.len() != expected_len {
        return Err("batch semantic response count mismatch".into());
    }
    let mut ordered = vec![None; expected_len];
    for item in response.results {
        if item.index >= expected_len || ordered[item.index].is_some() {
            return Err("batch semantic response index mismatch".into());
        }
        ordered[item.index] = Some(SemanticModelOutput {
            category: item.category,
            summary: item.summary,
            entities: item.entities,
            relationships: item.relationships,
            confidence: item.confidence,
        });
    }
    ordered
        .into_iter()
        .map(|item| item.ok_or_else(|| "batch semantic response missing item".into()))
        .collect()
}

impl LlamaCppProvider {
    fn socket_address(host: &str, port: u16) -> Result<SocketAddr, String> {
        (host, port)
            .to_socket_addrs()
            .map_err(|error| format!("invalid llama.cpp endpoint {host}:{port}: {error}"))?
            .next()
            .ok_or_else(|| format!("llama.cpp endpoint {host}:{port} unavailable"))
    }

    fn is_port_reachable(host: &str, port: u16) -> bool {
        Self::socket_address(host, port)
            .map(|address| TcpStream::connect_timeout(&address, Duration::from_millis(250)).is_ok())
            .unwrap_or(false)
    }

    pub fn chat_reachable(&self) -> bool {
        Self::is_port_reachable(&self.host, self.chat_port)
    }
    pub fn embed_reachable(&self) -> bool {
        Self::is_port_reachable(&self.host, self.embed_port)
    }

    /// Starts the chat/vision `llama-server` if the binary and model files
    /// are present and it isn't already listening. Returns `Ok(None)` (not
    /// an error) when setup isn't complete yet — capture and the rest of
    /// Chronicle must keep working with local AI simply pending setup.
    pub fn start_chat_server_if_needed(&self) -> Result<Option<Child>, String> {
        if self.chat_reachable() {
            return Ok(None);
        }
        if !engine_paths::runtime_installed() || !engine_paths::chat_model_installed() {
            return Ok(None);
        }
        let (Some(chat_model), Some(mmproj)) = (engine_paths::chat_model(), engine_paths::mmproj())
        else {
            return Ok(None);
        };
        let (out, err) = open_server_log("chat-server");
        Command::new(engine_paths::server_binary())
            .args(chat_server_args(&chat_model, &mmproj, &self.host, self.chat_port))
            .stdout(out)
            .stderr(err)
            .spawn()
            .map(Some)
            .map_err(|error| format!("unable to start the chat/vision engine: {error}"))
    }

    /// Starts the embedding `llama-server` if the binary and model file are
    /// present and it isn't already listening. Same "pending, not failed"
    /// behavior as `start_chat_server_if_needed` when setup isn't complete.
    pub fn start_embed_server_if_needed(&self) -> Result<Option<Child>, String> {
        if self.embed_reachable() {
            return Ok(None);
        }
        if !engine_paths::runtime_installed() || !engine_paths::embed_model_installed() {
            return Ok(None);
        }
        let Some(embed_model) = engine_paths::embed_model() else {
            return Ok(None);
        };
        let (out, err) = open_server_log("embed-server");
        Command::new(engine_paths::server_binary())
            .args(embed_server_args(&embed_model, &self.host, self.embed_port))
            .stdout(out)
            .stderr(err)
            .spawn()
            .map(Some)
            .map_err(|error| format!("unable to start the embedding engine: {error}"))
    }

    pub fn status(&self) -> LocalModelStatus {
        LocalModelStatus {
            chat_endpoint: format!("http://{}:{}", self.host, self.chat_port),
            embedding_endpoint: format!("http://{}:{}", self.host, self.embed_port),
            chat_model: self.chat_model.clone(),
            embedding_model: self.embedding_model.clone(),
            chat_available: self.chat_reachable(),
            embedding_available: self.embed_reachable(),
        }
    }

    /// Runs a chat prompt through the in-process native engine
    /// (`native_inference`) rather than an HTTP call to `llama-server`. Text,
    /// embeddings, and vision (`analyze_image`, via `native_inference`'s
    /// `mtmd`-backed `VisionEngine`) all run this way now — nothing in this
    /// provider still talks to `llama-server` over HTTP.
    fn generate_chat(&self, prompt: &str, max_tokens: u32) -> Result<String, String> {
        let chat_model = engine_paths::chat_model().ok_or("no data directory configured")?;
        if !chat_model.is_file() {
            return Err("chat model not installed".into());
        }
        // Request only as much context as this specific prompt plausibly
        // needs (roughly 4 chars/token, plus the response budget) instead
        // of always asking for the full `SERVER_CONTEXT_SIZE` — a short
        // single-event prompt gets a much smaller, cheaper-to-allocate KV
        // cache than an 8-item numbered batch does. `safe_context_size`
        // (inside `generation_engine`) still has the final say and can
        // shrink this further if memory is tight.
        let estimated_prompt_tokens = (prompt.len() as u32 / 4).saturating_add(max_tokens);
        let requested_ctx = crate::memory_planner::context_size_for_tokens(estimated_prompt_tokens);
        let n_ctx = std::num::NonZeroU32::new(requested_ctx).expect("context_size_for_tokens is always nonzero");
        let engine = crate::native_inference::generation_engine(&chat_model, n_ctx, inference_thread_count() as i32)?;
        engine.generate(prompt, max_tokens)
    }

    #[allow(dead_code)]
    pub fn analyze_text(&self, input: &str) -> Result<SemanticModelOutput, String> {
        let prompt = format!("Return JSON only with category, summary, entities, relationships, confidence (0..1). Interpret:\n{input}");
        let content = self.generate_chat(&prompt, MAX_RESPONSE_TOKENS)?;
        parse_and_validate_model_json(&content)
    }

    /// Analyze several contexts in one chat request. The indexed response
    /// prevents an item from being silently assigned to the wrong event.
    /// This is the same numbered-prompt technique used with every backend
    /// this provider has had — it's a prompting strategy, not something the
    /// server needs to support natively, since chat completion APIs don't
    /// offer "batch of independent prompts" as a primitive.
    pub fn analyze_text_batch(
        &self,
        inputs: &[String],
    ) -> Result<Vec<SemanticModelOutput>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let numbered = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| format!("ITEM {index}:\n{input}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let prompt = format!("Return JSON only as {{\"results\":[{{\"index\":0,\"category\":\"...\",\"summary\":\"...\",\"entities\":[],\"relationships\":[],\"confidence\":0.0}}]}}. Include exactly one result for every item, preserving its index.\n{numbered}");
        let content = self.generate_chat(&prompt, MAX_RESPONSE_TOKENS.saturating_mul(inputs.len() as u32))?;
        parse_batch_response(&content, inputs.len())
    }

    /// Runs a screenshot through the in-process native vision engine
    /// (`native_inference::vision_engine`, backed by `llama-cpp-2`'s `mtmd`
    /// feature) instead of base64-posting it to a separately spawned
    /// `llama-server --mmproj`. Same prompt contract every other backend
    /// this provider has used for vision, just decoded through
    /// `MtmdContext::tokenize`/`eval_chunks` rather than an HTTP
    /// `image_url` message.
    pub fn analyze_image(&self, bytes: &[u8]) -> Result<SemanticModelOutput, String> {
        validate_image_input(bytes)?;
        let chat_model = engine_paths::chat_model().ok_or("no data directory configured")?;
        let mmproj = engine_paths::mmproj().ok_or("no data directory configured")?;
        if !chat_model.is_file() || !mmproj.is_file() {
            return Err("chat/vision model not installed".into());
        }
        let n_ctx = std::num::NonZeroU32::new(SERVER_CONTEXT_SIZE).expect("SERVER_CONTEXT_SIZE is nonzero");
        let engine = crate::native_inference::vision_engine(
            &chat_model,
            &mmproj,
            n_ctx,
            inference_thread_count() as i32,
        )?;
        let prompt = "Return JSON only with category, summary, entities, relationships, confidence (0..1). Interpret this screenshot.";
        let content = engine.generate_with_image(bytes, prompt, MAX_RESPONSE_TOKENS)?;
        parse_and_validate_model_json(&content)
    }

    /// Embeds a batch of inputs in one in-process call via
    /// `native_inference::EmbeddingEngine`, which packs every input into the
    /// same context as its own sequence and reads back one pooled vector
    /// per sequence — no HTTP round trip and no separately running
    /// `llama-server` needed for embeddings at all.
    pub fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let embed_model = engine_paths::embed_model().ok_or("no data directory configured")?;
        if !embed_model.is_file() {
            return Err("embedding model not installed".into());
        }
        let n_ctx = std::num::NonZeroU32::new(SERVER_CONTEXT_SIZE).expect("SERVER_CONTEXT_SIZE is nonzero");
        let engine =
            crate::native_inference::embedding_engine(&embed_model, n_ctx, inference_thread_count() as i32)?;
        engine.embed_batch(inputs)
    }
}
impl LocalSemanticAnalyzer for LlamaCppProvider {
    fn analyze_text(&self, input: &str) -> Result<SemanticModelOutput, String> {
        self.analyze_text(input)
    }
    fn analyze_image(&self, bytes: &[u8]) -> Result<SemanticModelOutput, String> {
        self.analyze_image(bytes)
    }
}
impl TextEmbedder for LlamaCppProvider {
    fn dimensions(&self) -> usize {
        768
    }
    fn embed(&self, input: &str) -> Result<Vec<f32>, String> {
        self.embed_batch(&[input.to_string()])?
            .into_iter()
            .next()
            .ok_or("embedding engine returned no embedding".into())
    }
}
/// `LlamaCppProvider::default()` reads `CHRONICLE_LLAMA_*` env vars, which
/// are process-global. Tests that set them (to point the provider at a mock
/// server) and tests that assert on the unset defaults would otherwise race
/// when `cargo test` runs them on different threads of the same process.
/// This lock — shared with `asynchronous_processing_queue`'s end-to-end test
/// — makes any such env-var-touching test mutually exclusive.
#[cfg(test)]
pub(crate) fn env_var_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(test)]
#[path = "tests/local_model_provider_tests.rs"]
mod tests;
