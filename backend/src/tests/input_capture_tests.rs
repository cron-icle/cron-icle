    use super::*;
    #[test]
    fn mouse_events_store_coordinates_without_text() {
        let event =
            normalize_mouse_event("mouse_click", 10, 20, Some("left"), Some("Editor".into()));
        assert_eq!(event.source, "mouse_hook");
        assert_eq!(event.text, None);
        assert!(event.metadata_json.contains("10"));
    }
    #[test]
    fn keyboard_metadata_event_does_not_require_text() {
        let event = normalize_keyboard_event("key_down", 65, Some("Editor".into()), None);
        assert_eq!(event.privacy_class, "input_metadata");
        assert_eq!(event.text, None);
    }
    #[test]
    fn text_batcher_clamps_debounce_and_preserves_order() {
        let mut batcher = MetadataTextBatcher::default();
        batcher.push("a");
        batcher.push("b");
        assert!(batcher.flush_if_due(Duration::ZERO).is_none());
        assert!(batcher.flush_if_due(MAX_TEXT_BATCH_DEBOUNCE).is_none());
    }
    #[test]
    fn keyboard_text_requires_explicit_allowlisted_application() {
        let settings = InputCaptureSettings {
            capture_keyboard_text: true,
            keyboard_text_allowlist: vec!["Editor".into()],
            ..Default::default()
        };
        assert!(settings.allows_keyboard_text("editor"));
        assert!(!settings.allows_keyboard_text("Browser"));
    }
    #[test]
    fn normalization_drops_text_for_non_allowlisted_application() {
        let settings = InputCaptureSettings {
            capture_keyboard_text: true,
            keyboard_text_allowlist: vec!["Editor".into()],
            ..Default::default()
        };
        let event = normalize_allowlisted_keyboard_event(
            &settings,
            "key_down",
            65,
            Some("Browser".into()),
            Some("secret".into()),
        );
        assert!(event.text.is_none());
    }
    #[test]
    fn legacy_input_settings_have_no_text_allowlist() {
        let settings: InputCaptureSettings = serde_json::from_str(r#"{"mouse_enabled":false,"keyboard_enabled":true,"capture_keyboard_text":false,"excluded_applications":[]}"#).unwrap();
        assert!(settings.keyboard_text_allowlist.is_empty());
        assert!(!settings.allows_keyboard_text("Editor"));
    }
