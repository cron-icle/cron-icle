mod data_directory;
#[allow(dead_code)]
mod activity_capture;
#[allow(dead_code)]
mod asynchronous_processing_queue;
mod capture_writer;
#[allow(dead_code)]
mod embedding_provider;
mod filesystem_activity_capture;
#[allow(dead_code)]
mod input_capture;
mod local_inference_setup;
mod hardware_profiler;
mod local_model_provider;
mod memory_planner;
mod native_inference;
#[allow(dead_code)]
mod local_semantic_processing;
mod local_sqlite_event_database;
mod tauri_application_commands;
#[allow(dead_code)]
mod transient_screenshot_capture;
mod windows_active_window_screenshot;
mod windows_graphics_capture_session;
#[allow(dead_code)]
mod windows_ui_automation_capture;

use tauri::Manager;
use tauri_application_commands::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter("chronicle=info")
        .with_target(false)
        .init();

    // Never blocks: if the user hasn't chosen a data directory yet (see
    // `data_directory`), `AppState::initialize` below runs in a temporary,
    // non-persistent mode instead of prompting here. The user picks one from
    // Settings when they set up local AI.
    match data_directory::current() {
        Some(dir) => tracing::info!(path = %dir.display(), "using data directory"),
        None => tracing::info!("no data directory chosen yet; running in temporary mode until Settings is used to pick one"),
    }

    // Database open failures are recoverable: AppState::initialize() falls
    // back to a transient in-memory database and records the failure so the
    // UI can surface it, rather than panicking the whole process.
    let state = AppState::initialize();
    tauri::Builder::default()
        .manage(state)
        .setup(|app| {
            let state = app.state::<AppState>();
            let capture_enabled = state
                .settings
                .lock()
                .map(|settings| settings.enabled)
                .unwrap_or(false);
            if capture_enabled {
                tauri_application_commands::start_capture_state(&state)
                    .map_err(|error| Box::<dyn std::error::Error>::from(error))?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            tauri_application_commands::health_check,
            tauri_application_commands::capture_active_window_screenshot,
            tauri_application_commands::graphics_capture_session_available,
            tauri_application_commands::recent_event_count,
            tauri_application_commands::list_events,
            tauri_application_commands::list_semantic_events,
            tauri_application_commands::list_raw_event_processing_overview,
            tauri_application_commands::record_event,
            tauri_application_commands::record_semantic_event,
            tauri_application_commands::semantic_for_event,
            tauri_application_commands::delete_all_data,
            tauri_application_commands::get_capture_settings,
            tauri_application_commands::update_capture_settings,
            tauri_application_commands::export_data,
            tauri_application_commands::start_capture,
            tauri_application_commands::stop_capture,
            tauri_application_commands::capture_status,
            tauri_application_commands::set_input_permission,
            tauri_application_commands::set_screenshot_permission,
            tauri_application_commands::set_keyboard_text_allowlist,
            tauri_application_commands::set_excluded_applications,
            tauri_application_commands::set_excluded_paths,
            tauri_application_commands::set_watched_folders,
            tauri_application_commands::processing_queue_status,
            tauri_application_commands::processing_metrics,
            tauri_application_commands::inference_telemetry,
            tauri_application_commands::storage_usage,
            tauri_application_commands::model_provider_status,
            tauri_application_commands::processing_queue_limits,
            tauri_application_commands::capture_diagnostics,
            tauri_application_commands::cancel_pending_processing_tasks,
            tauri_application_commands::retry_failed_processing_tasks,
            tauri_application_commands::processing_status_for_event,
            tauri_application_commands::startup_diagnostics,
            tauri_application_commands::get_data_directory,
            tauri_application_commands::change_data_directory,
            local_inference_setup::local_ai_setup_status,
            local_inference_setup::cancel_model_download,
            local_inference_setup::setup_download_chat_model,
            local_inference_setup::setup_download_embed_model,
            local_inference_setup::setup_start_engine,
            local_inference_setup::setup_remove_chat_model,
            local_inference_setup::setup_remove_embed_model
        ])
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                let state = window.app_handle().state::<AppState>();
                tauri_application_commands::stop_capture_state(&state);
                tauri_application_commands::shutdown_llama_engine(&state);
            }
        })
        .run(tauri::generate_context!())
        .unwrap_or_else(|error| {
            tracing::error!(%error, "Chronicle exited because the Tauri event loop failed to start");
            std::process::exit(1);
        });
}
