    use super::*;

    #[test]
    fn defaults_use_local_engine_ports() {
        let _guard = env_var_lock().lock().unwrap();
        let p = LlamaCppProvider::default();
        assert_eq!(p.host, "127.0.0.1");
        assert!(p.chat_port > 0);
        assert!(p.embed_port > 0);
        assert_ne!(p.chat_port, p.embed_port);
        assert!(!p.chat_model.is_empty());
        assert!(!p.embedding_model.is_empty());
    }

    #[test]
    fn chat_server_args_enable_jinja_and_wide_context() {
        let args = chat_server_args(
            Path::new("model.gguf"),
            Path::new("mmproj.gguf"),
            "127.0.0.1",
            8090,
        );
        assert!(
            args.contains(&"--jinja".to_string()),
            "chat server must run with --jinja so Gemma 3's chat template is applied; \
             without it /v1/chat/completions fails while the port stays reachable, \
             which is exactly the silent-failure mode reported: {args:?}"
        );
        let ctx_index = args.iter().position(|a| a == "-c").expect("-c flag present");
        assert_eq!(args[ctx_index + 1], SERVER_CONTEXT_SIZE.to_string());
    }

    #[test]
    fn embed_server_args_include_wide_context() {
        let args = embed_server_args(Path::new("embed.gguf"), "127.0.0.1", 8091);
        assert!(args.contains(&"--embeddings".to_string()));
        let ctx_index = args.iter().position(|a| a == "-c").expect("-c flag present");
        assert_eq!(args[ctx_index + 1], SERVER_CONTEXT_SIZE.to_string());
    }

    #[test]
    fn server_args_pin_an_explicit_thread_count() {
        let expected = inference_thread_count().to_string();
        let chat_args = chat_server_args(Path::new("m.gguf"), Path::new("mm.gguf"), "127.0.0.1", 8090);
        let chat_index = chat_args.iter().position(|a| a == "-t").expect("-t flag present");
        assert_eq!(chat_args[chat_index + 1], expected);

        let embed_args = embed_server_args(Path::new("e.gguf"), "127.0.0.1", 8091);
        let embed_index = embed_args.iter().position(|a| a == "-t").expect("-t flag present");
        assert_eq!(embed_args[embed_index + 1], expected);
    }

    #[test]
    fn inference_thread_count_leaves_a_core_for_the_rest_of_the_app() {
        let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let threads = inference_thread_count();
        assert!(threads >= 1);
        if available > 1 {
            assert_eq!(threads, available - 1);
        } else {
            assert_eq!(threads, 1);
        }
    }

    fn provider_for(port: u16) -> LlamaCppProvider {
        LlamaCppProvider {
            host: "127.0.0.1".into(),
            chat_port: port,
            embed_port: port,
            chat_model: "test-chat".into(),
            embedding_model: "test-embed".into(),
        }
    }

    /// 1x1 transparent PNG — the minimal input `validate_image_input`
    /// accepts (its PNG-signature check, not real image content).
    const TEST_PNG: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21,
        196, 137, 0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13, 10, 45, 180, 0, 0, 0, 0, 73,
        69, 78, 68, 174, 66, 96, 130,
    ];

    // `analyze_text`/`analyze_text_batch`/`embed_batch` now run in-process
    // via `native_inference` (see `generate_chat`), not over HTTP, so they
    // can't be exercised with `mock_http_server` any more — and must not be,
    // since going through the real `engine_paths`-resolved model path in a
    // unit test can pick up a real, already-downloaded multi-gigabyte model
    // on a developer machine that has actually set up local AI (the global,
    // OS-level data-directory pointer `data_directory::current()` reads
    // isn't test-scoped). Their HTTP-transport coverage below now targets
    // `analyze_image`, the one method still on the HTTP path (vision isn't
    // ported to the native engine yet); `analyze_text_batch`'s
    // response-parsing/reordering contract is covered separately via the
    // pure `parse_batch_response` below; real native-engine coverage lives
    // in `native_inference`'s env-var-gated tests against a real GGUF model.

    #[test]
    fn analyze_image_bounds_generation_with_max_tokens() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let (port, handle) = mock_http_server(move |request| {
            sender.send(request).unwrap();
            let content = serde_json::json!({
                "category": "coding",
                "summary": "s",
                "entities": [],
                "relationships": [],
                "confidence": 0.5
            })
            .to_string();
            serde_json::json!({"choices": [{"message": {"content": content}}]}).to_string()
        });
        let provider = provider_for(port);
        provider.analyze_image(TEST_PNG).expect("analyze_image should succeed");
        handle.join().unwrap();
        let request: serde_json::Value = serde_json::from_str(&receiver.recv().unwrap()).unwrap();
        let max_tokens = request["max_tokens"]
            .as_u64()
            .expect("request must bound generation with max_tokens, or a bad generation can pin a worker thread indefinitely");
        assert_eq!(max_tokens, MAX_RESPONSE_TOKENS as u64);
    }

    #[test]
    fn analyze_image_parses_llama_server_chat_response() {
        let (port, handle) = mock_http_server(|_request| {
            let content = serde_json::json!({
                "category": "coding",
                "summary": "Editing Rust source",
                "entities": ["chronicle"],
                "relationships": [],
                "confidence": 0.9
            })
            .to_string();
            serde_json::json!({"choices": [{"message": {"content": content}}]}).to_string()
        });
        let provider = provider_for(port);
        let result = provider.analyze_image(TEST_PNG).expect("analyze_image should succeed");
        assert_eq!(result.category, "coding");
        assert_eq!(result.summary, "Editing Rust source");
        handle.join().unwrap();
    }

    #[test]
    fn chat_completion_surfaces_engine_errors_instead_of_panicking() {
        let provider = provider_for(1);
        let err = provider
            .analyze_image(TEST_PNG)
            .expect_err("unreachable port must error, not panic");
        assert!(err.contains("unavailable") || err.contains("engine"));
    }

    #[test]
    fn parse_batch_response_reorders_by_response_index() {
        let content = serde_json::json!({
            "results": [
                {"index": 1, "category": "b", "summary": "second", "entities": [], "relationships": [], "confidence": 0.5},
                {"index": 0, "category": "a", "summary": "first", "entities": [], "relationships": [], "confidence": 0.5}
            ]
        })
        .to_string();
        let results = parse_batch_response(&content, 2).expect("batch should parse");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].summary, "first");
        assert_eq!(results[1].summary, "second");
    }

    #[test]
    fn parse_batch_response_rejects_mismatched_result_count() {
        let content = serde_json::json!({
            "results": [
                {"index": 0, "category": "a", "summary": "only one", "entities": [], "relationships": [], "confidence": 0.5}
            ]
        })
        .to_string();
        let err = parse_batch_response(&content, 2)
            .expect_err("count mismatch must error, not silently misassign results");
        assert!(err.contains("count mismatch"));
    }

    #[test]
    fn parse_batch_response_rejects_out_of_range_or_duplicate_index() {
        let content = serde_json::json!({
            "results": [
                {"index": 0, "category": "a", "summary": "first", "entities": [], "relationships": [], "confidence": 0.5},
                {"index": 0, "category": "b", "summary": "dup", "entities": [], "relationships": [], "confidence": 0.5}
            ]
        })
        .to_string();
        let err = parse_batch_response(&content, 2).expect_err("duplicate index must error");
        assert!(err.contains("index mismatch"));
    }

    #[test]
    fn embed_batch_returns_empty_for_empty_input_without_touching_the_engine() {
        let provider = provider_for(1);
        assert_eq!(provider.embed_batch(&[]).unwrap(), Vec::<Vec<f32>>::new());
    }
