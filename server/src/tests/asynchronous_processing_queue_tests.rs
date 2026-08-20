    use super::*;
    use crate::local_sqlite_event_database::RawEvent;
    use std::sync::atomic::AtomicUsize;
    #[test]
    fn retries_back_off() {
        assert_eq!(MAX_RETRY_ATTEMPTS, 3);
        assert!(retry_delay(2) > retry_delay(1));
        assert_eq!(retry_delay(0), Duration::from_millis(250));
    }

    #[test]
    fn provider_panics_are_convertible_to_failures() {
        let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
            panic!("model failure")
        }));
        assert!(result.is_err());
    }

    #[test]
    fn processing_metrics_start_empty() {
        let mut metrics = ProcessingMetrics::default();
        metrics.record_completed_with_latency(Duration::from_millis(25));
        metrics.record_failed();
        metrics.record_panicked();
        metrics.record_model("test-model", "1");
        assert_eq!(metrics.average_latency_ms(), Some(25.0));
        assert_eq!(
            metrics.snapshot(),
            ProcessingMetrics {
                completed: 1,
                failed: 1,
                panicked: 1,
                total_latency_ms: 25,
                last_model_name: Some("test-model".into()),
                last_model_version: Some("1".into())
            }
        );
        metrics.reset();
        assert_eq!(metrics, ProcessingMetrics::default());
        assert_eq!(metrics.average_latency_ms(), None);
    }

    #[test]
    fn busy_worker_processes_bounded_work_and_stops() {
        struct BusyProcessor {
            calls: AtomicUsize,
        }
        impl QueueTaskProcessor for BusyProcessor {
            fn process(&self, _task: &QueueTask) -> Result<(), String> {
                std::thread::sleep(Duration::from_millis(10));
                self.calls.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }
        let database = Arc::new(Mutex::new(Database::in_memory().unwrap()));
        database
            .lock()
            .unwrap()
            .insert_event(&RawEvent {
                id: "busy-event".into(),
                timestamp_ns: 1,
                event_type: "test".into(),
                source: "test".into(),
                app_name: None,
                executable_path: None,
                process_id: None,
                window_handle: None,
                window_title: None,
                element_name: None,
                text: None,
                file_path: None,
                metadata_json: "{}".into(),
                privacy_class: "test".into(),
                confidence: 1.0,
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .unwrap();
        database
            .lock()
            .unwrap()
            .enqueue_task(&QueueTask {
                id: "busy-task".into(),
                raw_event_id: "busy-event".into(),
                task_type: TaskType::SemanticTextAnalysis,
                status: QueueStatus::Pending,
                attempts: 0,
                priority: 0,
            })
            .unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let processor = Arc::new(BusyProcessor {
            calls: AtomicUsize::new(0),
        });
        let worker = run_processing_worker(database.clone(), stop.clone(), processor.clone());
        std::thread::sleep(Duration::from_millis(5));
        database
            .lock()
            .unwrap()
            .insert_event(&RawEvent {
                id: "capture-while-busy".into(),
                timestamp_ns: 2,
                event_type: "window_focused".into(),
                source: "foreground_window".into(),
                app_name: Some("Editor".into()),
                executable_path: None,
                process_id: None,
                window_handle: None,
                window_title: Some("Notes".into()),
                element_name: None,
                text: None,
                file_path: None,
                metadata_json: "{}".into(),
                privacy_class: "metadata".into(),
                confidence: 1.0,
                created_at: "2026-01-01T00:00:01Z".into(),
            })
            .unwrap();
        let processing_started = std::time::Instant::now();
        std::thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);
        worker.join().unwrap();
        assert!(processing_started.elapsed() < Duration::from_secs(2));
        assert_eq!(processor.calls.load(Ordering::Relaxed), 1);
        assert_eq!(database.lock().unwrap().count_events().unwrap(), 2);
        assert_eq!(
            database
                .lock()
                .unwrap()
                .queue_counts()
                .unwrap()
                .get("complete"),
            Some(&1)
        );
    }

    /// `run_processing_worker` (the wrapper with no metrics observer) must
    /// still behave identically to the metrics-tracking version for callers
    /// that don't care about metrics — this is what production startup used
    /// before `processing_metrics` existed, and every other test in this
    /// module still calls it, so it must keep working unchanged.
    #[test]
    fn metrics_are_recorded_for_both_successes_and_failures() {
        struct FlakyProcessor {
            fail_next: std::sync::atomic::AtomicBool,
        }
        impl QueueTaskProcessor for FlakyProcessor {
            fn process(&self, _task: &QueueTask) -> Result<(), String> {
                if self.fail_next.swap(false, Ordering::Relaxed) {
                    Err("simulated engine failure".into())
                } else {
                    Ok(())
                }
            }
        }
        let database = Arc::new(Mutex::new(Database::in_memory().unwrap()));
        // Different task types so the worker processes them in separate
        // batches (claim_next_tasks only pulls additional work of the same
        // type as the first-claimed task) — otherwise both would land in
        // one process_batch call and the default `QueueTaskProcessor::
        // process_batch` short-circuits on the first error, which would
        // fail both tasks together instead of exercising one success and
        // one failure independently.
        for (id, event_id, task_type) in [
            ("fail-task", "fail-event", TaskType::SemanticTextAnalysis),
            ("ok-task", "ok-event", TaskType::EmbeddingGeneration),
        ] {
            database
                .lock()
                .unwrap()
                .insert_event(&RawEvent {
                    id: event_id.into(),
                    timestamp_ns: 1,
                    event_type: "test".into(),
                    source: "test".into(),
                    app_name: None,
                    executable_path: None,
                    process_id: None,
                    window_handle: None,
                    window_title: None,
                    element_name: None,
                    text: None,
                    file_path: None,
                    metadata_json: "{}".into(),
                    privacy_class: "test".into(),
                    confidence: 1.0,
                    created_at: "2026-01-01T00:00:00Z".into(),
                })
                .unwrap();
            database
                .lock()
                .unwrap()
                .enqueue_task(&QueueTask {
                    id: id.into(),
                    raw_event_id: event_id.into(),
                    task_type,
                    status: QueueStatus::Pending,
                    attempts: 0,
                    priority: 0,
                })
                .unwrap();
        }
        let stop = Arc::new(AtomicBool::new(false));
        let processor = Arc::new(FlakyProcessor {
            fail_next: std::sync::atomic::AtomicBool::new(true),
        });
        let metrics = Arc::new(Mutex::new(ProcessingMetrics::default()));
        let worker =
            run_processing_worker_with_metrics(database.clone(), stop.clone(), processor, metrics.clone());

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let snapshot = metrics.lock().unwrap().snapshot();
            if snapshot.completed >= 1 && snapshot.failed >= 1 {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "metrics were not updated in time: {snapshot:?}");
            std::thread::sleep(Duration::from_millis(20));
        }
        stop.store(true, Ordering::Relaxed);
        worker.join().unwrap();

        let snapshot = metrics.lock().unwrap().snapshot();
        assert_eq!(snapshot.completed, 1, "successful task must be counted");
        assert_eq!(snapshot.failed, 1, "failed task must be counted separately from successes");
        assert_eq!(snapshot.panicked, 0);
        assert!(snapshot.average_latency_ms().is_some(), "completed work must record latency");
    }

    /// Drives the real, production `LocalModelQueueProcessor` (not a test
    /// double) through the full path a captured event actually takes:
    /// insert raw event -> enqueue -> `run_processing_worker` claims it ->
    /// `LlamaCppProvider` calls the real in-process `native_inference`
    /// engine -> semantic result and embedding land back in SQLite.
    ///
    /// Unlike the HTTP-era version of this test, this can't be pointed at a
    /// mock server: `generate_chat`/`embed_batch` resolve model files via
    /// `engine_paths`, which reads a real, process-global, OS-level data
    /// directory pointer (`data_directory::current()`, cached in a
    /// `OnceLock` for the lifetime of the test binary) — there is no
    /// per-test override seam for it. So this test is real end-to-end or
    /// nothing: it runs for real, against whatever chat/embedding models are
    /// actually installed, and skips (rather than failing) when none are —
    /// keeping it green on machines without local AI set up while still
    /// giving genuine full-pipeline coverage on ones that do. See
    /// `native_inference`'s own gated tests for engine-level coverage
    /// against small test models, and `local_model_provider`'s
    /// `analyze_image_*` tests for the still-HTTP vision path.
    ///
    /// `#[ignore]`d because a full-size chat model (Gemma 3 4B) genuinely
    /// takes minutes per attempt on CPU, and — a real, tracked limitation,
    /// not a test-tuning issue — the native path has no grammar-constrained
    /// JSON decoding yet (see the `JSON_GRAMMAR` comment in
    /// `native_inference.rs` for why: a hand-written GBNF grammar crashed
    /// this llama.cpp build's grammar engine outright, so it was reverted
    /// rather than shipped). Without that constraint a real model can
    /// produce non-JSON prose past `max_tokens`, and each failed parse
    /// costs a full retry — measured up to and past 600s end-to-end on this
    /// hardware. Run explicitly with `cargo test -- --ignored` on a machine
    /// with local AI installed when validating the real model boundary;
    /// it's not part of the default fast suite.
    #[test]
    #[ignore = "slow: real multi-minute CPU inference against an installed multi-gigabyte model"]
    fn full_pipeline_processes_event_end_to_end_with_installed_models() {
        use crate::local_model_provider::engine_paths;
        if !engine_paths::chat_model_installed() || !engine_paths::embed_model_installed() {
            eprintln!("skipping: no local AI models installed on this machine");
            return;
        }

        let database = Arc::new(Mutex::new(Database::in_memory().unwrap()));
        database
            .lock()
            .unwrap()
            .insert_event(&RawEvent {
                id: "e2e-event".into(),
                timestamp_ns: 1,
                event_type: "window_focused".into(),
                source: "foreground_window".into(),
                app_name: Some("VS Code".into()),
                executable_path: None,
                process_id: None,
                window_handle: None,
                window_title: Some("local_model_provider.rs".into()),
                element_name: None,
                text: Some("editing the llama server integration".into()),
                file_path: None,
                metadata_json: "{}".into(),
                privacy_class: "content".into(),
                confidence: 1.0,
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .unwrap();
        database
            .lock()
            .unwrap()
            .enqueue_task(&QueueTask {
                id: "e2e-task".into(),
                raw_event_id: "e2e-event".into(),
                task_type: TaskType::SemanticTextAnalysis,
                status: QueueStatus::Pending,
                attempts: 0,
                priority: 0,
            })
            .unwrap();

        let processor = Arc::new(LocalModelQueueProcessor {
            database: database.clone(),
            screenshot_cache: Arc::new(Mutex::new(ScreenshotCache::new(16))),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let worker = run_processing_worker(database.clone(), stop.clone(), processor);

        // Each lookup is its own statement (not chained inside the `if let`
        // scrutinee) so its `MutexGuard` temporary drops at the semicolon
        // instead of living for the whole `if let` body — chaining a second
        // `.lock()` inside that body would try to lock the same `Mutex`
        // while the first guard was still alive and deadlock immediately.
        // Real model load + real CPU generation, not a mock — a full-size
        // Gemma 3 4B chat model generating up to MAX_RESPONSE_TOKENS on CPU
        // can legitimately take several minutes, so this deadline is
        // generous on purpose. This test is opt-in (gated on installed
        // models), not part of the fast default loop.
        let deadline = std::time::Instant::now() + Duration::from_secs(600);
        let semantic = loop {
            let semantic = database
                .lock()
                .unwrap()
                .semantic_for_raw_event("e2e-event")
                .unwrap();
            let completed = database
                .lock()
                .unwrap()
                .queue_counts()
                .unwrap()
                .get("complete")
                .copied()
                .unwrap_or(0);
            if let Some(semantic) = semantic {
                if completed >= 2 {
                    break semantic;
                }
            }
            if std::time::Instant::now() > deadline {
                stop.store(true, Ordering::Relaxed);
                worker.join().unwrap();
                panic!("event was not fully processed (semantic analysis + embedding) within the deadline");
            }
            std::thread::sleep(Duration::from_millis(200));
        };

        stop.store(true, Ordering::Relaxed);
        worker.join().unwrap();

        // The model's actual reading of the event is real, unmocked output
        // now, so this only asserts the pipeline produced *some* valid
        // structured result and stored it correctly — not specific content.
        assert!(!semantic.category.is_empty());
        assert!(!semantic.summary.is_empty());
        assert!(
            database.lock().unwrap().embedding_exists(&semantic.id).unwrap(),
            "embedding produced by the worker must be persisted"
        );
    }
