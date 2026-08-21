    use super::*;

    /// These tests need a real (tiny) GGUF model on disk and are slow
    /// (model load + generation), so they're gated behind an env var rather
    /// than run on every `cargo test`. Point `CRONICLE_TEST_GGUF_MODEL` at
    /// a small instruction-tuned or embedding GGUF (e.g. a `stories260K`-
    /// style test model for generation, or a small embedding model) to run
    /// them locally; CI without that env var skips them rather than failing.
    fn test_model_path() -> Option<std::path::PathBuf> {
        std::env::var("CRONICLE_TEST_GGUF_MODEL").ok().map(std::path::PathBuf::from)
    }

    /// Companion mmproj GGUF for `test_model_path`, needed only by the
    /// `VisionEngine` test below. Point `CRONICLE_TEST_MMPROJ_MODEL` at the
    /// multimodal projector matching whatever vision-capable chat model
    /// `CRONICLE_TEST_GGUF_MODEL` points at.
    fn test_mmproj_path() -> Option<std::path::PathBuf> {
        std::env::var("CRONICLE_TEST_MMPROJ_MODEL").ok().map(std::path::PathBuf::from)
    }

    /// 1x1 transparent PNG — the minimal input `VisionEngine` can decode
    /// through `MtmdBitmap::from_buffer` (a real, if content-free, PNG).
    const TEST_PNG: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21,
        196, 137, 0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13, 10, 45, 180, 0, 0, 0, 0, 73,
        69, 78, 68, 174, 66, 96, 130,
    ];

    #[test]
    fn vision_engine_produces_nonempty_output_from_a_real_model() {
        let (Some(model_path), Some(mmproj_path)) = (test_model_path(), test_mmproj_path()) else {
            eprintln!("skipping: CRONICLE_TEST_GGUF_MODEL and/or CRONICLE_TEST_MMPROJ_MODEL not set");
            return;
        };
        let engine = VisionEngine::load(&model_path, &mmproj_path, 0, NonZeroU32::new(4096).unwrap(), 2)
            .expect("vision model + mmproj should load");
        let output = engine
            .generate_with_image(TEST_PNG, "Describe this image.", 32)
            .expect("generation should succeed");
        assert!(!output.is_empty(), "model produced no output");
    }

    #[test]
    fn generation_engine_produces_nonempty_output_from_a_real_model() {
        let Some(path) = test_model_path() else {
            eprintln!("skipping: CRONICLE_TEST_GGUF_MODEL not set");
            return;
        };
        let engine = GenerationEngine::load(&path, 0, NonZeroU32::new(2048).unwrap(), 2)
            .expect("model should load");
        let output = engine.generate("Say hello.", 32).expect("generation should succeed");
        assert!(!output.is_empty(), "model produced no output");
    }

    #[test]
    fn generation_engine_lifecycle_loads_stays_resident_and_unloads_when_idle() {
        let Some(path) = test_model_path() else {
            eprintln!("skipping: CRONICLE_TEST_GGUF_MODEL not set");
            return;
        };
        let n_ctx = NonZeroU32::new(2048).unwrap();

        // First call loads (Unloaded -> Ready).
        generation_engine(&path, n_ctx, 2).expect("first load should succeed");
        assert_eq!(generation_engine_state(), ModelState::Ready);

        // A second call to the same model path reuses the resident engine
        // rather than reloading — reflected in load time, not just result,
        // but at minimum it must not error and must keep reporting Ready.
        generation_engine(&path, n_ctx, 2).expect("cached call should succeed");
        assert_eq!(generation_engine_state(), ModelState::Ready);

        // Sweeping while fresh (within keep-alive) must not unload.
        sweep_idle_engines();
        assert_eq!(generation_engine_state(), ModelState::Ready, "must not unload while within keep-alive");

        // Backdate last_used past the keep-alive window and confirm the
        // engine is reported Idle, then actually freed on the next sweep —
        // the real behavior the 60s `GENERATION_KEEP_ALIVE` constant would
        // produce, without a test actually waiting 60 seconds.
        force_generation_engine_stale_for_test();
        assert_eq!(generation_engine_state(), ModelState::Idle);
        sweep_idle_engines();
        assert_eq!(generation_engine_state(), ModelState::Unloaded, "idle sweep must actually free the engine");

        // And it must be transparently reloadable after being unloaded.
        generation_engine(&path, n_ctx, 2).expect("reload after unload should succeed");
        assert_eq!(generation_engine_state(), ModelState::Ready);

        // A later request for MORE context than the resident engine has
        // must reload with the larger size rather than silently reusing a
        // too-small context (see `generation_engine`'s doc comment) — this
        // is the case adaptive per-request context sizing
        // (`local_model_provider::generate_chat`) actually depends on.
        let bigger_ctx = NonZeroU32::new(4096).unwrap();
        let engine = generation_engine(&path, bigger_ctx, 2).expect("upgrade reload should succeed");
        assert!(engine.n_ctx() >= bigger_ctx, "must reload with at least the newly requested context size");

        // A request for LESS context than the (now-larger) resident engine
        // has must reuse it rather than downgrade-reload.
        let engine = generation_engine(&path, n_ctx, 2).expect("smaller request should reuse resident engine");
        assert!(engine.n_ctx() >= bigger_ctx, "a smaller request must not shrink the resident context");
    }

    #[test]
    fn embedding_engine_produces_consistent_dimension_vectors() {
        let Some(path) = test_model_path() else {
            eprintln!("skipping: CRONICLE_TEST_GGUF_MODEL not set");
            return;
        };
        let engine = EmbeddingEngine::load(&path, 0, NonZeroU32::new(2048).unwrap(), 2)
            .expect("model should load");
        let embeddings = engine
            .embed_batch(&["first input".into(), "second input".into()])
            .expect("embedding should succeed");
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), embeddings[1].len());
        assert!(!embeddings[0].is_empty());
    }
