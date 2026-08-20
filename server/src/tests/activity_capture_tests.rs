    use super::*;
    #[test]
    fn normalized_event_has_evidence() {
        let e = normalize_window_event("Editor".into(), "main.rs".into(), None, Some(7));
        assert_eq!(e.event_type, "window_focused");
        assert_eq!(e.process_id, Some(7));
        assert!(!e.id.is_empty());
    }
    #[test]
    fn defaults_are_privacy_safe() {
        let s = CaptureSettings::default();
        assert!(!s.enabled);
        assert!(matches!(s.keyboard_mode, KeyboardMode::MetadataOnly));
        assert!(!s.screenshots_enabled);
    }

    #[test]
    fn path_exclusions_match_whole_path_segments_case_insensitively() {
        let settings = CaptureSettings {
            excluded_paths: vec!["secrets".into()],
            ..Default::default()
        };
        assert!(settings.excludes_path("C:\\Projects\\Secrets\\notes.txt"));
        assert!(!settings.excludes_path("C:\\Projects\\Public\\notes.txt"));
    }

    #[test]
    fn path_exclusions_do_not_over_match_partial_segments() {
        let settings = CaptureSettings {
            excluded_paths: vec!["secrets".into()],
            ..Default::default()
        };
        // "Secretariat" contains "secret" as a substring but is not the
        // "secrets" path segment, so it must not be excluded.
        assert!(!settings.excludes_path("C:\\Projects\\Secretariat\\notes.txt"));
    }

    #[test]
    fn application_exclusion_matches_exact_executable_name_only() {
        let settings = CaptureSettings {
            excluded_applications: vec!["code".into()],
            ..Default::default()
        };
        assert!(settings.excludes_application("C:\\Tools\\Code.exe", "Code"));
        assert!(!settings.excludes_application("C:\\Tools\\decode.exe", "decode"));
        assert!(!settings.excludes_application("C:\\Tools\\Encoder.exe", "Encoder"));
    }

    #[test]
    fn application_exclusion_with_extension_matches_full_filename() {
        let settings = CaptureSettings {
            excluded_applications: vec!["code.exe".into()],
            ..Default::default()
        };
        assert!(settings.excludes_application("C:\\Tools\\Code.exe", "Code"));
        assert!(!settings.excludes_application("C:\\Tools\\decode.exe", "decode"));
    }

    #[test]
    fn legacy_settings_default_path_exclusions_to_empty() {
        let settings: CaptureSettings = serde_json::from_str(r#"{"enabled":true,"mouse_enabled":false,"keyboard_enabled":false,"keyboard_mode":"metadata_only","excluded_applications":[],"watched_folders":[],"screenshots_enabled":false}"#).unwrap();
        assert!(settings.excluded_paths.is_empty());
        assert!(settings.keyboard_text_allowlist.is_empty());
    }
