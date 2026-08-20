    use super::*;
    #[test]
    fn assets_are_in_memory_and_expire() {
        let asset = TransientScreenshotAsset::new("event".into(), vec![1, 2, 3], "image/png");
        assert_eq!(asset.bytes.len(), 3);
        assert!(!asset.expired(Duration::from_secs(1)));
    }
    #[test]
    fn meaningful_triggers_are_explicit() {
        assert!(ScreenshotTrigger::DoubleClick.meaningful());
        assert!(ScreenshotTrigger::ElementFocused.meaningful());
    }
    #[test]
    fn store_releases_assets_after_processing() {
        let mut store = TransientScreenshotStore::default();
        assert!(store.insert(TransientScreenshotAsset::new(
            "event".into(),
            vec![1],
            "image/png"
        )));
        assert_eq!(store.len(), 1);
        assert!(store.take("event").is_some());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn store_associates_asset_with_queue_task() {
        let mut store = TransientScreenshotStore::default();
        store.insert(TransientScreenshotAsset::new(
            "event".into(),
            vec![1],
            "image/png",
        ));
        assert!(store.associate_queue_task("event", "task".into()));
        assert_eq!(
            store.take("event").unwrap().queue_task_id.as_deref(),
            Some("task")
        );
    }

    #[test]
    fn store_purges_expired_assets() {
        let mut store = TransientScreenshotStore::default();
        store.insert(TransientScreenshotAsset::new(
            "expired".into(),
            vec![1],
            "image/png",
        ));
        store.purge_expired(Duration::ZERO);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn store_rejects_empty_and_non_image_assets() {
        let mut store = TransientScreenshotStore::default();
        assert!(!store.insert(TransientScreenshotAsset::new(
            "empty".into(),
            vec![],
            "image/png"
        )));
        assert!(!store.insert(TransientScreenshotAsset::new(
            "text".into(),
            vec![1],
            "text/plain"
        )));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn default_retention_is_short_and_explicit() {
        assert_eq!(DEFAULT_SCREENSHOT_RETENTION, Duration::from_secs(30));
    }

    #[test]
    fn dispatcher_queues_only_meaningful_triggers() {
        let mut dispatcher = ScreenshotTriggerDispatcher::default();
        dispatcher.request("event", ScreenshotTrigger::DoubleClick);
        assert_eq!(
            dispatcher.drain(),
            vec![("event".into(), ScreenshotTrigger::DoubleClick)]
        );
        assert!(dispatcher.drain().is_empty());
    }

    #[test]
    fn graphics_capture_probe_is_safe_for_invalid_handle() {
        assert!(!graphics_capture_item_available(0));
    }
