    use super::*;

    #[test]
    fn detect_reports_at_least_one_logical_core() {
        let profile = HardwareProfile::detect();
        assert!(profile.logical_cores >= 1);
    }

    #[test]
    fn detect_never_panics_and_gpu_is_honestly_absent() {
        // The real assertion here is that this doesn't panic — hardware
        // detection running during app startup must never be a crash risk.
        let profile = HardwareProfile::detect();
        assert!(profile.gpu.is_none(), "no GPU backend is wired in yet; reporting one would be a lie");
    }

    #[cfg(windows)]
    #[test]
    fn detect_reports_nonzero_ram_on_windows() {
        let profile = HardwareProfile::detect();
        assert!(profile.total_ram_bytes > 0, "GlobalMemoryStatusEx should succeed on any real Windows machine");
        assert!(profile.available_ram_bytes <= profile.total_ram_bytes);
    }
