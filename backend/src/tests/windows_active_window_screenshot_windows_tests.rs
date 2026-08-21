    use super::*;
    #[test]
    fn invalid_window_handle_is_reported_without_panicking() {
        assert!(capture_window_png(0).is_err());
    }
