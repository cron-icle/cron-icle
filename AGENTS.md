# Cronicle agent rules

These rules apply to all work in this repository. They are intentionally strict because Cronicle observes sensitive computer activity and runs native Windows code.

## Product invariants

- Capture and raw persistence must remain useful when AI is disabled, slow, unavailable, or incorrect.
- Raw events are append-only evidence. Semantic interpretations must reference raw events and must never overwrite them.
- Capture work must never run on the UI thread and must never wait for model inference.
- Transient screenshots and other visual assets must be held in memory by default and released after processing.
- Keyboard capture is opt-in. Never persist passwords, credentials, banking/payment input, secure-desktop input, UAC input, or configured excluded applications.
- Do not add cloud processing, telemetry, automatic updates, browser extensions, or unrestricted filesystem scanning without explicit product direction.

## Daemon architecture

Cronicle ships as a single standalone binary, not a packaged desktop app: one process owns capture, SQLite, local LLM inference, and an embedded HTTP server (axum) that serves both a JSON API and the built React frontend on `127.0.0.1`. There is no installer, no code-signing/notarization step, and no native app shell — the binary auto-opens the user's browser to the UI on launch and keeps running headless regardless of whether that tab stays open. See `backend/src/http/` for the route table and `backend/src/lib.rs` for server startup/shutdown.

- Keep the frontend in `frontend/src/` and native/backend code in `backend/src/`.
- Use Rust for Windows integration, capture providers, persistence, queue workers, and privacy enforcement.
- Keep HTTP handlers in `backend/src/http/*.rs` thin: extract the request, delegate to a plain function in `app_service.rs`/`inference/setup.rs` (transport-agnostic business logic — no `axum`/HTTP types), and serialize the result. Business-logic functions take `&AppState` (or `Arc<AppState>` where a handler needs to move it into a spawned task) and return `Result<T, String>`; they must stay callable from something other than an HTTP handler (a future CLI, a test) without modification.
- There is no push/event channel to the browser (no Tauri `emit`, no WebSocket by default) — long-running operations (model downloads, data-directory moves) run fire-and-forget on a spawned task and report progress through a pollable `AppState` field + `GET` endpoint instead. Don't reach for a blocking request/response for anything that can take more than a second or two.
- Name modules by responsibility, not vague names. Prefer `capture::activity`, `persistence::sqlite`, `processing::queue`, and `app_service` over `capture`, `db`, `queue`, or `commands`.
- Organize platform-specific Rust providers as folders with a shared `mod.rs` contract, `windows.rs` for native Win32/WinRT integration, and `portable.rs` for the shared macOS+Linux implementation (see `capture/activity/`, `capture/input/`, `capture/screenshot/`, `capture/active_window/`, `capture/graphics_session/`). Prefer one `portable.rs` over separate `mac.rs`/`linux.rs` files when a single well-maintained crate (`xcap`, `rdev`, `active-win-pos-rs`) already covers both — only split them if their real implementations diverge enough to need it. `capture/ui_automation/` is the current exception: it still has no macOS/Linux implementation (no established crate for cross-platform accessibility-tree reads), so its `portable.rs` is an honest `Ok(None)` stub — see README's Known limitations before assuming this module works off Windows. Keep Windows API calls out of shared contracts.
- Add module-level Rustdoc for every native module describing ownership, threading, privacy, and failure behavior.
- Use typed structs/enums for event types, queue statuses, settings, and command payloads. Avoid unvalidated stringly-typed state when an enum is practical.
- Keep Windows-only APIs behind `cfg(windows)` and macOS/Linux-only APIs behind their respective `cfg(target_os = ...)`, with a safe fallback for any other target so the crate still compiles/tests everywhere.
- Never use `unwrap()` in capture loops, command handlers, or worker threads. Convert failures into logged, non-fatal results where capture can continue.
- Do not hold a database mutex across sleeps, Windows API waits, model calls, or filesystem operations.
- All background threads need a stop signal and must be joined or safely detached during application shutdown.
- Prefer stable, well-maintained libraries over in-house reimplementations. Before hand-rolling something a crate already solves — filesystem watching, vector similarity search, HTTP clients, connection pooling, event hooks — use the established crate (e.g. `notify` for filesystem events, `sqlite-vec` for vector search, `r2d2`/`r2d2_sqlite` for connection pooling, `windows-rs` for Win32 APIs like `SetWinEventHook`) unless there is a documented, specific reason it does not fit. In-house implementations are a last resort, not a default, and any exception must be justified in a code comment.

## Database and migrations

- Use migrations for schema changes; never silently mutate production tables in application code.
- Preserve foreign keys and append-only raw evidence.
- Keep FTS/vector indexes rebuildable from source records.
- Add repository tests for every new insert, query, update, delete, retry, and migration path.
- Bound query limits and worker queues to prevent unbounded memory growth.

## Testing and verification

Before handing off a change, run the smallest relevant checks and then the full suite when practical:

```powershell
npm run test:frontend
npm test
npm run build
```

- Add unit tests for normalization, privacy filtering, retry behavior, database ordering, and failure handling.
- Add Windows integration tests for hooks and permissions when the implementation touches native APIs.
- Do not claim a native provider is complete when only its interface or normalizer exists.
- Update `README.md` when setup, commands, architecture, privacy behavior, or verification steps change.

## Git safety and commit conventions

- Never run `git commit`, `git push`, `git merge`, `git rebase`, `git tag`, or other history-changing commands unless the user gives an explicit command to do so in the current conversation.
- “Implement”, “finish”, “clean up”, or “prepare” does not authorize committing or pushing.
- Read `git status`, inspect the diff, and run relevant tests before a user-authorized commit.
- Never include unrelated user changes in a commit. Ask if the intended scope is ambiguous.
- Never force-push, reset hard, delete branches, or rewrite history unless explicitly requested.
- Use Conventional Commits with a required type and imperative subject:
  - `feat:` user-visible capability
  - `fix:` bug correction
  - `refactor:` behavior-preserving restructuring
  - `test:` tests only
  - `docs:` documentation only
  - `chore:` tooling/dependency/maintenance work
  - `perf:` performance improvement
  - `build:` build/release changes
- Keep the subject concise, lowercase after the prefix, and free of a trailing period.
- If a change has a breaking API or schema change, use `!` or a `BREAKING CHANGE:` footer.
- Report the exact commit hash and push result after an authorized Git operation.

## Communication

- State the active implementation task before editing.
- Report meaningful blockers with evidence and the smallest safe alternative.
- Be explicit about what is implemented, what is an interface only, and what remains machine-specific.
- For implementation requests, prefer completing one cohesive feature area end-to-end—backend contracts, UI integration, tests, and README updates—before moving to the next feature. Avoid substituting a series of unrelated micro-tasks for a larger requested feature.
