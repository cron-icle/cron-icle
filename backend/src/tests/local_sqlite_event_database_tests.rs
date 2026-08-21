    use super::*;

    fn event(id: &str, timestamp_ns: i64, title: &str, text: Option<&str>) -> RawEvent {
        RawEvent {
            id: id.into(),
            timestamp_ns,
            event_type: "window_focused".into(),
            source: "test".into(),
            app_name: Some("Test App".into()),
            executable_path: None,
            process_id: Some(42),
            window_handle: None,
            window_title: Some(title.into()),
            element_name: None,
            text: text.map(str::to_owned),
            file_path: None,
            metadata_json: "{}".into(),
            privacy_class: "safe".into(),
            confidence: 1.0,
            created_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn creates_schema_and_starts_empty() {
        let database = Database::in_memory().unwrap();
        assert_eq!(database.count_events().unwrap(), 0);
        let counts = database.storage_counts().unwrap();
        assert_eq!(counts.get("raw_events"), Some(&0));
        assert_eq!(counts.get("semantic_events"), Some(&0));
        assert_eq!(counts.get("embeddings"), Some(&0));
        assert_eq!(counts.get("queue_tasks"), Some(&0));
    }

    #[test]
    fn inserts_and_returns_newest_events_first() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("old", 10, "Older", None))
            .unwrap();
        database
            .insert_event(&event("new", 20, "Newer", None))
            .unwrap();
        let events = database.recent_events(10, None).unwrap();
        assert_eq!(
            events
                .iter()
                .map(|event| event.id.as_str())
                .collect::<Vec<_>>(),
            vec!["new", "old"]
        );
    }

    #[test]
    fn recovery_does_not_duplicate_pending_image_tasks() {
        let database = Database::in_memory().unwrap();
        let mut image_event = event("image-recovery", 1, "Image", None);
        image_event.window_handle = Some(123);
        database.insert_event(&image_event).unwrap();
        database
            .enqueue_task(&QueueTask {
                id: "image-task".into(),
                raw_event_id: image_event.id.clone(),
                task_type: TaskType::SemanticImageAnalysis,
                status: QueueStatus::Pending,
                attempts: 0,
                priority: 0,
            })
            .unwrap();
        assert_eq!(database.enqueue_unprocessed_events(10).unwrap(), 0);
        assert_eq!(database.queue_counts().unwrap().get("pending"), Some(&1));
    }

    #[test]
    fn window_events_use_text_processing_when_screenshots_are_disabled() {
        let database = Database::in_memory().unwrap();
        let mut window_event = event("screen-off", 1, "Window", None);
        window_event.window_handle = Some(123);
        database.insert_event_and_enqueue(&window_event).unwrap();
        let task = database.claim_next_task().unwrap().unwrap();
        assert_eq!(task.task_type, TaskType::SemanticTextAnalysis);
    }

    #[test]
    fn raw_event_listing_does_not_search_private_evidence() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event(
                "rust",
                10,
                "Rust compiler",
                Some("cargo test passed"),
            ))
            .unwrap();
        database
            .insert_event(&event(
                "notes",
                20,
                "Meeting notes",
                Some("project planning"),
            ))
            .unwrap();
        let results = database.recent_events(10, Some("compiler")).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn seed_is_idempotent() {
        let database = Database::in_memory().unwrap();
        database.seed_ready_event().unwrap();
        database.seed_ready_event().unwrap();
        assert_eq!(database.count_events().unwrap(), 1);
    }

    #[test]
    fn delete_all_removes_raw_and_derived_records() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("one", 10, "One", None))
            .unwrap();
        database.delete_all().unwrap();
        assert_eq!(database.count_events().unwrap(), 0);
        assert!(database.recent_events(10, Some("One")).unwrap().is_empty());
    }

    #[test]
    fn delete_all_removes_embeddings_before_parent_events() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("delete-embedded", 1, "Embedded", None))
            .unwrap();
        database
            .insert_semantic_event(&SemanticEvent {
                id: "semantic-delete".into(),
                raw_event_id: "delete-embedded".into(),
                category: "test".into(),
                summary: "embedded record".into(),
                entities_json: "[]".into(),
                relationships_json: "[]".into(),
                confidence: 1.0,
                model_name: "test".into(),
                model_version: "1".into(),
                created_at: "now".into(),
            })
            .unwrap();
        database
            .insert_embedding("semantic-delete", "test", "1", &[1.0, 0.0])
            .unwrap();
        database.delete_all().unwrap();
        let counts = database.storage_counts().unwrap();
        assert_eq!(counts.get("raw_events"), Some(&0));
        assert_eq!(counts.get("semantic_events"), Some(&0));
        assert_eq!(counts.get("embeddings"), Some(&0));
    }

    #[test]
    fn queue_claim_and_finish_round_trip() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("event-1", 1, "Queue source", None))
            .unwrap();
        let task = QueueTask {
            id: "task-1".into(),
            raw_event_id: "event-1".into(),
            task_type: TaskType::SemanticTextAnalysis,
            status: QueueStatus::Pending,
            attempts: 0,
            priority: 5,
        };
        database.enqueue_task(&task).unwrap();
        let claimed = database.claim_next_task().unwrap().unwrap();
        assert_eq!(claimed.id, "task-1");
        assert_eq!(claimed.status, QueueStatus::Processing);
        database.finish_task("task-1").unwrap();
        assert!(database.claim_next_task().unwrap().is_none());
    }

    #[test]
    fn failed_retry_persists_a_future_retry_timestamp() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("event-retry", 1, "Retry", None))
            .unwrap();
        database
            .enqueue_task(&QueueTask {
                id: "task-retry".into(),
                raw_event_id: "event-retry".into(),
                task_type: TaskType::SemanticTextAnalysis,
                status: QueueStatus::Pending,
                attempts: 0,
                priority: 0,
            })
            .unwrap();
        database.claim_next_task().unwrap().unwrap();
        database
            .fail_task("task-retry", "temporary failure", true, 3)
            .unwrap();
        let retry_at: Option<String> = database
            .connection
            .query_row(
                "SELECT retry_at FROM processing_queue WHERE id = 'task-retry'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(retry_at.is_some());
        assert!(database.claim_next_task().unwrap().is_none());
    }

    #[test]
    fn cancellation_marks_only_pending_tasks() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("event-cancel", 1, "Cancel", None))
            .unwrap();
        database
            .enqueue_task(&QueueTask {
                id: "task-cancel".into(),
                raw_event_id: "event-cancel".into(),
                task_type: TaskType::EmbeddingGeneration,
                status: QueueStatus::Pending,
                attempts: 0,
                priority: 0,
            })
            .unwrap();
        assert_eq!(database.cancel_pending_tasks().unwrap(), 1);
        assert_eq!(database.queue_counts().unwrap().get("cancelled"), Some(&1));
        assert!(database.claim_next_task().unwrap().is_none());
    }

    #[test]
    fn processing_tasks_can_be_requeued_on_shutdown() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("event-shutdown", 1, "Shutdown", None))
            .unwrap();
        database
            .enqueue_task(&QueueTask {
                id: "task-shutdown".into(),
                raw_event_id: "event-shutdown".into(),
                task_type: TaskType::EmbeddingGeneration,
                status: QueueStatus::Pending,
                attempts: 0,
                priority: 0,
            })
            .unwrap();
        database.claim_next_task().unwrap().unwrap();
        assert_eq!(database.requeue_processing_tasks().unwrap(), 1);
        assert!(database.claim_next_task().unwrap().is_some());
    }

    #[test]
    fn persists_one_thousand_events_without_losing_count() {
        let database = Database::in_memory().unwrap();
        let started = std::time::Instant::now();
        for index in 0..1_000 {
            database
                .insert_event(&event(&format!("bulk-{index}"), index, "Bulk", None))
                .unwrap();
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        assert_eq!(database.count_events().unwrap(), 1_000);
        assert_eq!(database.recent_events(10, None).unwrap().len(), 10);
    }

    #[test]
    fn fts_search_has_bounded_latency_at_one_thousand_events() {
        let database = Database::in_memory().unwrap();
        for index in 0..1_000 {
            database
                .insert_event(&event(
                    &format!("search-{index}"),
                    index,
                    if index == 777 {
                        "UniqueSearchMarker"
                    } else {
                        "Background"
                    },
                    None,
                ))
                .unwrap();
        }
        let started = std::time::Instant::now();
        let results = database
            .recent_events(25, Some("UniqueSearchMarker"))
            .unwrap();
        assert!(!results.is_empty());
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    #[test]
    fn stale_processing_tasks_are_requeued() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("event-stale", 1, "Stale", None))
            .unwrap();
        database
            .enqueue_task(&QueueTask {
                id: "task-stale".into(),
                raw_event_id: "event-stale".into(),
                task_type: TaskType::EmbeddingGeneration,
                status: QueueStatus::Pending,
                attempts: 0,
                priority: 0,
            })
            .unwrap();
        let claimed = database.claim_next_task().unwrap().unwrap();
        assert_eq!(claimed.status, QueueStatus::Processing);
        database.connection.execute("UPDATE processing_queue SET started_at = datetime('now', '-20 minutes') WHERE id = 'task-stale'", []).unwrap();
        assert_eq!(database.recover_stale_processing_tasks(10).unwrap(), 1);
        assert!(database.claim_next_task().unwrap().is_some());
    }

    #[test]
    fn semantic_event_requires_existing_raw_event() {
        let database = Database::in_memory().unwrap();
        let semantic = SemanticEvent {
            id: "semantic-1".into(),
            raw_event_id: "missing".into(),
            category: "test".into(),
            summary: "summary".into(),
            entities_json: "[]".into(),
            relationships_json: "[]".into(),
            confidence: 0.9,
            model_name: "test-model".into(),
            model_version: "1".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        assert!(database.insert_semantic_event(&semantic).is_err());
    }

    #[test]
    fn semantic_search_uses_summary_and_updates_fts_rows() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("event-search", 1, "Source", None))
            .unwrap();
        database
            .insert_semantic_event(&SemanticEvent {
                id: "semantic-search".into(),
                raw_event_id: "event-search".into(),
                category: "coding".into(),
                summary: "Reviewed compiler output".into(),
                entities_json: "[]".into(),
                relationships_json: "[]".into(),
                confidence: 1.0,
                model_name: "test".into(),
                model_version: "1".into(),
                created_at: "now".into(),
            })
            .unwrap();
        assert_eq!(
            database
                .recent_semantic_events(10, Some("compiler"))
                .unwrap()
                .len(),
            1
        );
        database.connection.execute("UPDATE semantic_events SET summary = 'Reviewed design notes' WHERE id = 'semantic-search'", []).unwrap();
        assert!(database
            .recent_semantic_events(10, Some("compiler"))
            .unwrap()
            .is_empty());
        assert_eq!(
            database
                .recent_semantic_events(10, Some("design"))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn hybrid_rank_combines_text_and_vector_results() {
        let database = Database::in_memory().unwrap();
        let ranked = database.hybrid_rank(
            &["text-only".into(), "shared".into()],
            &[("vector-only".into(), 0.9), ("shared".into(), 0.8)],
            3,
        );
        assert_eq!(ranked[0], "shared");
        assert_eq!(ranked.len(), 3);
    }

    #[test]
    fn embedding_fallback_search_ranks_similar_vectors() {
        let database = Database::in_memory().unwrap();
        database
            .insert_event(&event("event-embed", 1, "Embedding source", None))
            .unwrap();
        database
            .insert_semantic_event(&SemanticEvent {
                id: "semantic-embed".into(),
                raw_event_id: "event-embed".into(),
                category: "test".into(),
                summary: "vector".into(),
                entities_json: "[]".into(),
                relationships_json: "[]".into(),
                confidence: 1.0,
                model_name: "test".into(),
                model_version: "1".into(),
                created_at: "now".into(),
            })
            .unwrap();
        database
            .insert_embedding("semantic-embed", "test", "1", &[1.0, 0.0])
            .unwrap();
        assert_eq!(
            database.search_embeddings(&[0.9, 0.1], 1).unwrap()[0].0,
            "semantic-embed"
        );
    }
