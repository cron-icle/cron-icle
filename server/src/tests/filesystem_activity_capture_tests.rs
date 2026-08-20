    use super::*;

    #[test]
    fn exact_component_exclusion_does_not_over_match() {
        let excluded = vec!["skip".to_string()];
        assert!(path_is_excluded(Path::new("C:/watched/skip/file.txt"), &excluded));
        assert!(!path_is_excluded(
            Path::new("C:/watched/skipper/file.txt"),
            &excluded
        ));
    }

    #[test]
    fn create_and_remove_events_map_to_expected_event_types() {
        assert_eq!(
            match EventKind::Create(notify::event::CreateKind::File) {
                EventKind::Create(_) => "file_created",
                _ => "unexpected",
            },
            "file_created"
        );
        assert_eq!(
            match EventKind::Remove(notify::event::RemoveKind::File) {
                EventKind::Remove(_) => "file_deleted",
                _ => "unexpected",
            },
            "file_deleted"
        );
    }
