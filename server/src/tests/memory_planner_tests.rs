    use super::*;

    fn profile_with_available(bytes: u64) -> HardwareProfile {
        HardwareProfile {
            logical_cores: 8,
            total_ram_bytes: bytes.max(1),
            available_ram_bytes: bytes,
            gpu: None,
        }
    }

    #[test]
    fn estimate_grows_with_context_size_and_model_size() {
        let small_ctx = estimate_required_bytes(1_000_000_000, 1024);
        let large_ctx = estimate_required_bytes(1_000_000_000, 8192);
        assert!(large_ctx > small_ctx);

        let small_model = estimate_required_bytes(1_000_000_000, 4096);
        let large_model = estimate_required_bytes(4_000_000_000, 4096);
        assert!(large_model > small_model);
    }

    #[test]
    fn plan_load_picks_the_full_context_when_memory_is_plentiful() {
        let profile = profile_with_available(64 * 1024 * 1024 * 1024); // 64 GiB
        let plan = plan_load(3_000_000_000, 8192, &profile).expect("should fit comfortably");
        assert_eq!(plan.context_size, 8192);
    }

    #[test]
    fn plan_load_steps_down_context_when_memory_is_tight() {
        // Enough for the model plus a small context, not the full one.
        let model_bytes = 3_000_000_000u64;
        let available = estimate_required_bytes(model_bytes, 2048) + 50_000_000;
        // Scale available back up past the safety margin so 2048 just fits.
        let available = (available as f64 / 0.8) as u64;
        let profile = profile_with_available(available);
        let plan = plan_load(model_bytes, 8192, &profile).expect("a smaller context should still fit");
        assert!(plan.context_size <= 2048, "expected a stepped-down context, got {}", plan.context_size);
    }

    #[test]
    fn plan_load_refuses_when_nothing_fits() {
        let profile = profile_with_available(10_000_000); // 10 MB — nowhere near enough
        assert_eq!(plan_load(3_000_000_000, 8192, &profile), None);
    }

    #[test]
    fn plan_load_refuses_when_memory_reading_is_unavailable() {
        let profile = profile_with_available(0);
        assert_eq!(plan_load(1_000_000, 1024, &profile), None);
    }

    #[test]
    fn requested_context_size_caps_the_ladder() {
        let profile = profile_with_available(64 * 1024 * 1024 * 1024);
        let plan = plan_load(3_000_000_000, 2048, &profile).expect("should fit");
        assert_eq!(plan.context_size, 2048, "must not exceed what was actually requested");
    }

    #[test]
    fn adaptive_batch_size_scales_down_as_available_ram_shrinks() {
        assert_eq!(adaptive_batch_size(8, &profile_with_available(16 * 1024 * 1024 * 1024)), 8);
        assert_eq!(adaptive_batch_size(8, &profile_with_available(5 * 1024 * 1024 * 1024)), 4);
        assert_eq!(adaptive_batch_size(8, &profile_with_available(3 * 1024 * 1024 * 1024)), 1);
        assert_eq!(adaptive_batch_size(8, &profile_with_available(500 * 1024 * 1024)), 1);
    }

    #[test]
    fn adaptive_batch_size_never_exceeds_the_requested_max_or_drops_to_zero() {
        let plentiful = profile_with_available(64 * 1024 * 1024 * 1024);
        assert_eq!(adaptive_batch_size(3, &plentiful), 3, "must not scale above what was asked for");
        assert!(adaptive_batch_size(1, &profile_with_available(0)) >= 1, "must always allow at least one item");
    }

    #[test]
    fn context_size_for_tokens_picks_the_smallest_sufficient_rung() {
        assert_eq!(context_size_for_tokens(500), 1024);
        assert_eq!(context_size_for_tokens(1024), 1024);
        assert_eq!(context_size_for_tokens(1500), 2048);
        assert_eq!(context_size_for_tokens(5000), 8192);
    }

    #[test]
    fn context_size_for_tokens_falls_back_to_the_largest_rung_when_nothing_fits() {
        assert_eq!(context_size_for_tokens(100_000), 8192);
    }
