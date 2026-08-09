    use super::*;
    #[test]
    fn normalizes_element_metadata_without_screenshot() {
        let event = normalize_focused_element(FocusedElementSnapshot {
            element_name: Some("Save".into()),
            control_type: Some("Button".into()),
            ..Default::default()
        });
        assert_eq!(event.event_type, "element_focused");
        assert_eq!(event.source, "windows_ui_automation");
        assert!(event.metadata_json.contains("Button"));
    }

    #[test]
    fn bounds_selected_text_and_control_values() {
        let snapshot = FocusedElementSnapshot {
            selected_text: Some("x".repeat(5000)),
            element_value: Some("y".repeat(5000)),
            ..Default::default()
        }
        .sanitize();
        assert_eq!(snapshot.selected_text.unwrap().len(), 4096);
        assert_eq!(snapshot.element_value.unwrap().len(), 4096);
    }

    #[test]
    fn password_controls_never_retain_values() {
        let event = normalize_focused_element(FocusedElementSnapshot {
            control_type: Some("PasswordBox".into()),
            element_value: Some("secret".into()),
            selected_text: Some("secret".into()),
            ..Default::default()
        });
        assert_eq!(event.privacy_class, "protected_field");
        assert!(event.text.is_none());
        assert!(!event.metadata_json.contains("secret"));
    }

    #[test]
    fn unavailable_or_inaccessible_focus_returns_empty_result() {
        let provider = WindowsUiAutomationProvider;
        assert!(provider.focused_element().is_ok());
    }
