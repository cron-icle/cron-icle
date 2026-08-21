//! Runtime configuration read from the environment.

const DEFAULT_PORT: u16 = 47823;

/// The local-only port Cronicle binds its HTTP server to. Overridable via
/// `CRONICLE_PORT`, mainly so more than one instance can run side by side
/// during development.
pub fn resolve_port() -> u16 {
    std::env::var("CRONICLE_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

/// Whether to skip auto-opening the system browser on startup. Set by
/// integration tests and dev tooling that don't want a browser window
/// popping up on every run.
pub fn should_open_browser() -> bool {
    std::env::var("CRONICLE_NO_OPEN").is_err()
}
