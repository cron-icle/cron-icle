//! Local SQLite persistence for append-only raw evidence and derived records.
//!
//! The database is the reliability boundary: capture writes compact normalized
//! events first, while semantic processing may be retried or regenerated. FTS5
//! is maintained from raw-event triggers so search remains useful without AI.

use crate::activity_capture::CaptureSettings;
use crate::asynchronous_processing_queue::MAX_PENDING_TASKS;
use crate::asynchronous_processing_queue::{QueueStatus, QueueTask, TaskType};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Once;

/// Dimensionality of vectors produced by the configured local embedding model
/// (Nomic Embed Text). Embeddings of a different length are still stored in
/// the durable JSON/blob fallback columns but are skipped by the sqlite-vec
/// ANN index, which requires a fixed vector width per virtual table.
pub const EMBEDDING_DIMENSIONS: usize = 768;

static REGISTER_SQLITE_VEC: Once = Once::new();

/// Registers the `sqlite-vec` loadable extension with rusqlite's
/// auto-extension mechanism. Safe to call from multiple threads/connections;
/// registration only needs to happen once per process.
fn register_sqlite_vec_extension() {
    REGISTER_SQLITE_VEC.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEvent {
    pub id: String,
    pub timestamp_ns: i64,
    pub event_type: String,
    pub source: String,
    pub app_name: Option<String>,
    pub executable_path: Option<String>,
    pub process_id: Option<u32>,
    pub window_handle: Option<u64>,
    pub window_title: Option<String>,
    pub element_name: Option<String>,
    pub text: Option<String>,
    pub file_path: Option<String>,
    pub metadata_json: String,
    pub privacy_class: String,
    pub confidence: f32,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEvent {
    pub id: String,
    pub raw_event_id: String,
    pub category: String,
    pub summary: String,
    pub entities_json: String,
    pub relationships_json: String,
    pub confidence: f32,
    pub model_name: String,
    pub model_version: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEventView {
    pub id: String,
    pub raw_event_id: String,
    pub timestamp_ns: i64,
    pub app_name: Option<String>,
    pub window_title: Option<String>,
    pub category: String,
    pub summary: String,
    pub confidence: f32,
    pub model_name: String,
    pub created_at: String,
}

/// Pool of read-only connections used by UI query commands (list/search/
/// count/diagnostics). Kept separate from the single writer connection so
/// reads never block on, or block, writer traffic and capture threads never
/// wait behind UI queries for the database mutex.
pub type ReaderPool = r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>;

/// Builds the read-only connection pool against the same database file the
/// writer uses. Each pooled connection is set `query_only` so it structurally
/// cannot perform writes even if a caller passes it into a write path.
pub fn open_reader_pool() -> std::result::Result<ReaderPool, r2d2::Error> {
    // Must happen before any pooled connection is opened: `with_init` runs
    // its callback on an already-open handle, which is too late for
    // `sqlite3_auto_extension` to take effect on that connection.
    register_sqlite_vec_extension();
    let manager = r2d2_sqlite::SqliteConnectionManager::file(crate::data_directory::database_file()).with_init(
        |connection| {
            connection.pragma_update(None, "query_only", "ON")?;
            connection.pragma_update(None, "journal_mode", "WAL")?;
            Ok(())
        },
    );
    r2d2::Pool::builder().max_size(4).build(manager)
}

pub struct Database {
    connection: Connection,
}

impl Database {
    pub fn open() -> Result<Self> {
        // `sqlite3_auto_extension` only affects connections opened after
        // registration, so this must run before `Connection::open` — not
        // inside `from_connection`, which receives an already-open handle.
        register_sqlite_vec_extension();
        Self::from_connection(Connection::open(crate::data_directory::database_file())?)
    }

    fn from_connection(connection: Connection) -> Result<Self> {
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.execute_batch(include_str!("../migrations/001_initial.sql"))?;
        // Migrate pre-search builds away from the raw-event FTS surface.
        connection.execute_batch("DROP TRIGGER IF EXISTS raw_events_ai; DROP TRIGGER IF EXISTS raw_events_ad; DROP TABLE IF EXISTS raw_events_fts;")?;
        // Keep existing installations compatible with columns added after v1.
        add_column_if_missing(&connection, "processing_queue", "retry_at", "TEXT")?;
        // Binary vectors avoid JSON parsing overhead while retaining the JSON
        // column for backwards-compatible exports and older installations.
        add_column_if_missing(
            &connection,
            "semantic_event_embeddings",
            "embedding_blob",
            "BLOB",
        )?;
        // Real ANN/vector similarity search via the sqlite-vec loadable
        // extension. When the extension fails to register (e.g. platform
        // missing the bundled native library) this statement fails and we
        // fall back to the brute-force cosine scan in `search_embeddings`.
        if let Err(error) = connection.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_semantic_event_embeddings USING vec0(
                embedding float[{EMBEDDING_DIMENSIONS}],
                +semantic_event_id TEXT
            )"
        )) {
            tracing::warn!(%error, "sqlite-vec virtual table unavailable; using durable binary embedding storage fallback for vector search");
        }
        Ok(Self { connection })
    }

    /// Exposes the underlying connection for shared read-only query helpers
    /// (see `*_on` free functions) when no pooled reader connection is
    /// available and a read command must fall back to the writer.
    pub(crate) fn connection(&self) -> &Connection {
        &self.connection
    }

    fn vector_index_available(&self) -> bool {
        self.connection
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vec_semantic_event_embeddings'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .unwrap_or(None)
            .is_some()
    }

    #[cfg(test)]
    pub(crate) fn in_memory() -> Result<Self> {
        register_sqlite_vec_extension();
        Self::from_connection(Connection::open_in_memory()?)
    }

    /// Opens a transient in-memory database. Used as a recoverable fallback
    /// when the on-disk database fails to open, so a disk/permissions
    /// failure degrades capture (no persistence across restarts) instead of
    /// crashing the whole application.
    pub fn open_in_memory_degraded() -> Result<Self> {
        register_sqlite_vec_extension();
        Self::from_connection(Connection::open_in_memory()?)
    }

    pub fn count_events(&self) -> Result<i64> {
        count_events_on(&self.connection)
    }

    #[allow(dead_code)]
    pub fn storage_counts(&self) -> Result<HashMap<String, i64>> {
        storage_counts_on(&self.connection)
    }

    pub fn insert_event(&self, event: &RawEvent) -> Result<()> {
        self.connection.execute(
            "INSERT INTO raw_events (id, timestamp_ns, event_type, source, app_name, executable_path, process_id, window_handle, window_title, element_name, text, file_path, metadata_json, privacy_class, confidence, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![event.id, event.timestamp_ns, event.event_type, event.source, event.app_name, event.executable_path, event.process_id, event.window_handle.map(|value| value as i64), event.window_title, event.element_name, event.text, event.file_path, event.metadata_json, event.privacy_class, event.confidence, event.created_at],
        )?;
        Ok(())
    }

    pub fn insert_event_and_enqueue(&self, event: &RawEvent) -> Result<()> {
        self.insert_event(event)?;
        let task_type = if event.window_handle.is_some() && self.screenshots_enabled()? {
            TaskType::SemanticImageAnalysis
        } else {
            TaskType::SemanticTextAnalysis
        };
        self.enqueue_task(&QueueTask {
            id: uuid::Uuid::new_v4().to_string(),
            raw_event_id: event.id.clone(),
            task_type,
            status: QueueStatus::Pending,
            attempts: 0,
            priority: 0,
        })
    }

    /// Persists a batch of capture events (and enqueues each for semantic
    /// analysis) in one call. Used by the single capture-writer thread, which
    /// batches events from all capture sources so per-event database writes
    /// never happen on a hook/poller thread. `screenshots_enabled` is passed
    /// in (rather than re-read per event) since the writer already holds the
    /// current settings snapshot for the whole batch.
    pub fn insert_events_and_enqueue_batch(
        &self,
        events: &[RawEvent],
        screenshots_enabled: bool,
    ) -> Result<()> {
        for event in events {
            self.insert_event(event)?;
            let task_type = if event.window_handle.is_some() && screenshots_enabled {
                TaskType::SemanticImageAnalysis
            } else {
                TaskType::SemanticTextAnalysis
            };
            self.enqueue_task(&QueueTask {
                id: uuid::Uuid::new_v4().to_string(),
                raw_event_id: event.id.clone(),
                task_type,
                status: QueueStatus::Pending,
                attempts: 0,
                priority: 0,
            })?;
        }
        Ok(())
    }

    pub fn enqueue_unprocessed_events(&self, limit: u32) -> Result<usize> {
        let events = self.recent_events(limit, None)?;
        let mut queued = 0;
        for event in events {
            if self.semantic_for_raw_event(&event.id)?.is_some() {
                continue;
            }
            let task_type = if event.window_handle.is_some() && self.screenshots_enabled()? {
                TaskType::SemanticImageAnalysis
            } else {
                TaskType::SemanticTextAnalysis
            };
            let has_task: bool = self.connection.query_row("SELECT EXISTS(SELECT 1 FROM processing_queue WHERE raw_event_id = ?1 AND task_type IN ('semantic_text_analysis','SemanticTextAnalysis','semantic_image_analysis','SemanticImageAnalysis') AND status IN ('pending','processing'))", [&event.id], |row| row.get(0))?;
            if !has_task {
                self.enqueue_task(&QueueTask {
                    id: uuid::Uuid::new_v4().to_string(),
                    raw_event_id: event.id,
                    task_type,
                    status: QueueStatus::Pending,
                    attempts: 0,
                    priority: 0,
                })?;
                queued += 1;
            }
        }
        Ok(queued)
    }

    pub fn event_by_id(&self, id: &str) -> Result<Option<RawEvent>> {
        self.connection.query_row("SELECT id, timestamp_ns, event_type, source, app_name, executable_path, process_id, window_handle, window_title, element_name, text, file_path, metadata_json, privacy_class, confidence, created_at FROM raw_events WHERE id = ?1", [id], map_event).optional()
    }

    pub fn recent_events(&self, limit: u32, query: Option<&str>) -> Result<Vec<RawEvent>> {
        recent_events_on(&self.connection, limit, query)
    }

    #[allow(dead_code)]
    pub fn recent_semantic_events(
        &self,
        limit: u32,
        query: Option<&str>,
    ) -> Result<Vec<SemanticEventView>> {
        recent_semantic_events_on(&self.connection, limit, query)
    }

    pub fn delete_all(&self) -> Result<()> {
        self.connection.execute_batch(
            "DELETE FROM processing_queue; DELETE FROM semantic_event_embeddings; DELETE FROM semantic_events; DELETE FROM raw_events;",
        )
    }

    /// Merges the WAL file back into the main database file. Called before
    /// moving the data directory to a new location so the copy sees the
    /// database's actual contents rather than a stale main file plus a
    /// separate `-wal` file that a straight file copy would otherwise have
    /// to carry over just as faithfully.
    pub fn checkpoint_wal(&self) -> Result<()> {
        self.connection
            .pragma_update(None, "wal_checkpoint", "TRUNCATE")
    }

    pub fn save_setting(&self, key: &str, value_json: &str) -> Result<()> {
        self.connection.execute("INSERT INTO app_settings(key, value_json, updated_at) VALUES (?1, ?2, ?3) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at", params![key, value_json, Utc::now().to_rfc3339()])?;
        Ok(())
    }

    pub fn load_setting(&self, key: &str) -> Result<Option<String>> {
        self.connection
            .query_row(
                "SELECT value_json FROM app_settings WHERE key = ?1",
                [key],
                |row| row.get(0),
            )
            .optional()
    }

    fn screenshots_enabled(&self) -> Result<bool> {
        let Some(value) = self.load_setting("capture")? else {
            return Ok(false);
        };
        Ok(serde_json::from_str::<CaptureSettings>(&value)
            .map(|settings| settings.screenshots_enabled)
            .unwrap_or(false))
    }

    pub fn export_json(&self) -> Result<String> {
        let events = self.recent_events(100_000, None)?;
        serde_json::to_string_pretty(&HashMap::from([("events", events)]))
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))
    }

    #[allow(dead_code)]
    pub fn enqueue_task(&self, task: &QueueTask) -> Result<()> {
        let pending: i64 = self.connection.query_row(
            "SELECT COUNT(*) FROM processing_queue WHERE status = 'pending'",
            [],
            |row| row.get(0),
        )?;
        if pending >= MAX_PENDING_TASKS as i64 {
            return Err(rusqlite::Error::InvalidQuery);
        }
        self.connection.execute("INSERT INTO processing_queue (id, raw_event_id, task_type, status, priority, attempts, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'))", params![task.id, task.raw_event_id, serde_json::to_string(&task.task_type).unwrap_or_default().trim_matches('"'), serde_json::to_string(&task.status).unwrap_or_default().trim_matches('"'), task.priority, task.attempts])?;
        Ok(())
    }

    pub fn claim_next_task(&self) -> Result<Option<QueueTask>> {
        let transaction = self.connection.unchecked_transaction()?;
        let candidate = transaction.query_row("SELECT id, raw_event_id, task_type, attempts, priority FROM processing_queue WHERE status = 'pending' AND (retry_at IS NULL OR retry_at <= datetime('now')) ORDER BY priority DESC, created_at ASC LIMIT 1", [], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, u32>(3)?, row.get::<_, i32>(4)?))).optional()?;
        let Some((id, raw_event_id, task_type, attempts, priority)) = candidate else {
            transaction.commit()?;
            return Ok(None);
        };
        transaction.execute("UPDATE processing_queue SET status = 'processing', started_at = datetime('now'), attempts = attempts + 1 WHERE id = ?1", [&id])?;
        transaction.commit()?;
        let task_type = match task_type.as_str() {
            "SemanticTextAnalysis" | "semantic_text_analysis" => TaskType::SemanticTextAnalysis,
            "SemanticImageAnalysis" | "semantic_image_analysis" => TaskType::SemanticImageAnalysis,
            _ => TaskType::EmbeddingGeneration,
        };
        Ok(Some(QueueTask {
            id,
            raw_event_id,
            task_type,
            status: QueueStatus::Processing,
            attempts: attempts + 1,
            priority,
        }))
    }

    /// Claims at most `limit` pending tasks of one type in priority order.
    /// Keeping a batch homogeneous lets providers use one model request while
    /// preserving independent queue rows and retry state.
    pub fn claim_next_tasks(&self, task_type: &TaskType, limit: usize) -> Result<Vec<QueueTask>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let task_name = serde_json::to_string(task_type)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        let transaction = self.connection.unchecked_transaction()?;
        let mut statement = transaction.prepare("SELECT id, raw_event_id, attempts, priority FROM processing_queue WHERE status = 'pending' AND task_type = ?1 AND (retry_at IS NULL OR retry_at <= datetime('now')) ORDER BY priority DESC, created_at ASC LIMIT ?2")?;
        let rows = statement.query_map(params![task_name, limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, u32>(2)?,
                row.get::<_, i32>(3)?,
            ))
        })?;
        let candidates: Vec<_> = rows.collect::<rusqlite::Result<Vec<_>>>()?;
        drop(statement);
        for (id, _, _, _) in &candidates {
            transaction.execute("UPDATE processing_queue SET status = 'processing', started_at = datetime('now'), attempts = attempts + 1 WHERE id = ?1", [id])?;
        }
        transaction.commit()?;
        Ok(candidates
            .into_iter()
            .map(|(id, raw_event_id, attempts, priority)| QueueTask {
                id,
                raw_event_id,
                task_type: task_type.clone(),
                status: QueueStatus::Processing,
                attempts: attempts + 1,
                priority,
            })
            .collect())
    }

    pub fn finish_task(&self, task_id: &str) -> Result<()> {
        self.connection.execute("UPDATE processing_queue SET status = 'complete', completed_at = datetime('now') WHERE id = ?1", [task_id])?;
        Ok(())
    }
    pub fn fail_task(&self, task_id: &str, error: &str, retry: bool, attempt: u32) -> Result<()> {
        if retry {
            let retry_seconds =
                (250u64.saturating_mul(2u64.saturating_pow(attempt.min(8))) / 1000).max(1);
            self.connection.execute("UPDATE processing_queue SET status = 'pending', error = ?1, retry_at = datetime('now', '+' || ?2 || ' seconds'), completed_at = NULL WHERE id = ?3", params![error, retry_seconds.max(1) as i64, task_id])?;
        } else {
            self.connection.execute("UPDATE processing_queue SET status = 'failed', error = ?1, retry_at = NULL, completed_at = datetime('now') WHERE id = ?2", params![error, task_id])?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn queue_counts(&self) -> Result<HashMap<String, i64>> {
        queue_counts_on(&self.connection)
    }

    pub fn cancel_pending_tasks(&self) -> Result<usize> {
        Ok(self.connection.execute("UPDATE processing_queue SET status = 'cancelled', completed_at = datetime('now') WHERE status = 'pending'", [])?)
    }

    pub fn retry_failed_tasks(&self) -> Result<usize> {
        Ok(self.connection.execute("UPDATE processing_queue SET status = 'pending', retry_at = NULL, completed_at = NULL, error = NULL WHERE status = 'failed'", [])?)
    }

    pub fn requeue_processing_tasks(&self) -> Result<usize> {
        Ok(self.connection.execute("UPDATE processing_queue SET status = 'pending', started_at = NULL WHERE status = 'processing'", [])?)
    }

    #[allow(dead_code)]
    pub fn processing_status_for_raw_event(
        &self,
        raw_event_id: &str,
    ) -> Result<Vec<(String, String, u32, Option<String>)>> {
        processing_status_for_raw_event_on(&self.connection, raw_event_id)
    }

    pub fn recover_stale_processing_tasks(&self, stale_minutes: u32) -> Result<usize> {
        let changed = self.connection.execute("UPDATE processing_queue SET status = 'pending', started_at = NULL, error = 'requeued after interrupted processing' WHERE status = 'processing' AND started_at < datetime('now', ?1)", [format!("-{} minutes", stale_minutes)])?;
        Ok(changed)
    }

    pub fn insert_semantic_event(&self, event: &SemanticEvent) -> Result<()> {
        self.connection.execute("INSERT INTO semantic_events (id, raw_event_id, category, summary, entities_json, relationships_json, confidence, model_name, model_version, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)", params![event.id, event.raw_event_id, event.category, event.summary, event.entities_json, event.relationships_json, event.confidence, event.model_name, event.model_version, event.created_at])?;
        Ok(())
    }

    pub fn semantic_for_raw_event(&self, raw_event_id: &str) -> Result<Option<SemanticEvent>> {
        semantic_for_raw_event_on(&self.connection, raw_event_id)
    }

    #[allow(dead_code)]
    pub fn embedding_exists(&self, semantic_event_id: &str) -> Result<bool> {
        embedding_exists_on(&self.connection, semantic_event_id)
    }

    #[allow(dead_code)]
    pub fn insert_embedding(
        &self,
        semantic_event_id: &str,
        model_name: &str,
        model_version: &str,
        embedding: &[f32],
    ) -> Result<()> {
        // Durable binary/JSON storage remains the source of truth and the
        // fallback path when the sqlite-vec extension is unavailable.
        self.connection.execute("INSERT INTO semantic_event_embeddings (semantic_event_id, model_name, model_version, dimensions, embedding_json, embedding_blob, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now')) ON CONFLICT(semantic_event_id) DO UPDATE SET model_name=excluded.model_name, model_version=excluded.model_version, dimensions=excluded.dimensions, embedding_json=excluded.embedding_json, embedding_blob=excluded.embedding_blob, created_at=excluded.created_at", params![semantic_event_id, model_name, model_version, embedding.len() as i64, serde_json::to_string(embedding).unwrap_or_else(|_| "[]".into()), encode_embedding(embedding)])?;

        if self.vector_index_available() && embedding.len() == EMBEDDING_DIMENSIONS {
            if let Err(error) = self.upsert_vector_index(semantic_event_id, embedding) {
                tracing::warn!(%error, semantic_event_id, "failed to update sqlite-vec index; falling back to brute-force search for this embedding");
            }
        }
        Ok(())
    }

    fn upsert_vector_index(&self, semantic_event_id: &str, embedding: &[f32]) -> Result<()> {
        self.connection.execute(
            "DELETE FROM vec_semantic_event_embeddings WHERE semantic_event_id = ?1",
            params![semantic_event_id],
        )?;
        self.connection.execute(
            "INSERT INTO vec_semantic_event_embeddings (embedding, semantic_event_id) VALUES (?1, ?2)",
            params![vec_literal(embedding), semantic_event_id],
        )?;
        Ok(())
    }

    /// Vector similarity search. Prefers the sqlite-vec ANN index; falls back
    /// to an in-process brute-force cosine scan over the durable binary/JSON
    /// embedding columns when the extension could not be loaded or the query
    /// vector does not match the indexed dimensionality.
    #[allow(dead_code)]
    pub fn search_embeddings(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>> {
        if self.vector_index_available() && query.len() == EMBEDDING_DIMENSIONS {
            match self.search_embeddings_via_vec_index(query, limit) {
                Ok(results) => return Ok(results),
                Err(error) => {
                    tracing::warn!(%error, "sqlite-vec search failed; falling back to brute-force cosine scan");
                }
            }
        }
        self.search_embeddings_brute_force(query, limit)
    }

    fn search_embeddings_via_vec_index(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let mut statement = self.connection.prepare(
            "SELECT semantic_event_id, distance FROM vec_semantic_event_embeddings WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance",
        )?;
        let rows = statement.query_map(params![vec_literal(query), limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;
        // Convert distance (lower is closer) into a similarity-style score
        // (higher is closer) so callers can treat this the same as the
        // brute-force cosine path.
        rows.map(|row| row.map(|(id, distance)| (id, -(distance as f32))))
            .collect()
    }

    fn search_embeddings_brute_force(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let mut statement = self
            .connection
            .prepare("SELECT semantic_event_id, embedding_json, embedding_blob FROM semantic_event_embeddings")?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<Vec<u8>>>(2)?,
            ))
        })?;
        let mut scored = Vec::new();
        for row in rows {
            let (id, json, blob) = row?;
            let embedding = blob
                .and_then(|bytes| decode_embedding(&bytes))
                .unwrap_or_else(|| serde_json::from_str(&json).unwrap_or_default());
            if embedding.len() == query.len() {
                scored.push((id, cosine_similarity(query, &embedding)));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    #[allow(dead_code)]
    pub fn hybrid_rank(
        &self,
        text_ids: &[String],
        vector_scores: &[(String, f32)],
        limit: usize,
    ) -> Vec<String> {
        let text_rank: HashMap<&String, f32> = text_ids
            .iter()
            .enumerate()
            .map(|(index, id)| (id, 1.0 / (index as f32 + 1.0)))
            .collect();
        let vector_rank: HashMap<&String, f32> = vector_scores
            .iter()
            .map(|(id, score)| (id, *score))
            .collect();
        let mut ids: Vec<String> = text_ids
            .iter()
            .chain(vector_scores.iter().map(|(id, _)| id))
            .cloned()
            .collect();
        ids.sort();
        ids.dedup();
        ids.sort_by(|left, right| {
            let left_score = text_rank.get(left).copied().unwrap_or(0.0) * 0.4
                + vector_rank.get(left).copied().unwrap_or(0.0) * 0.6;
            let right_score = text_rank.get(right).copied().unwrap_or(0.0) * 0.4
                + vector_rank.get(right).copied().unwrap_or(0.0) * 0.6;
            right_score
                .partial_cmp(&left_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ids.truncate(limit);
        ids
    }

    pub fn seed_ready_event(&self) -> Result<()> {
        if self.count_events()? == 0 {
            self.insert_event(&RawEvent {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp_ns: Utc::now().timestamp_nanos_opt().unwrap_or_default(),
                event_type: "system_ready".into(),
                source: "chronicle".into(),
                app_name: Some("Chronicle".into()),
                executable_path: None,
                process_id: None,
                window_handle: None,
                window_title: Some("Desktop shell initialized".into()),
                element_name: None,
                text: None,
                file_path: None,
                metadata_json: "{}".into(),
                privacy_class: "safe".into(),
                confidence: 1.0,
                created_at: Utc::now().to_rfc3339(),
            })?;
        }
        Ok(())
    }
}

/// Read-only query implementations shared between the writer `Database` and
/// the read-only `r2d2` connection pool used by UI query commands. Both call
/// paths run the identical SQL against a plain `&Connection`, so pooled
/// reader connections never need to hold the writer mutex.
pub(crate) fn count_events_on(connection: &Connection) -> Result<i64> {
    connection.query_row("SELECT COUNT(*) FROM raw_events", [], |row| row.get(0))
}

pub(crate) fn storage_counts_on(connection: &Connection) -> Result<HashMap<String, i64>> {
    let mut counts = HashMap::new();
    for (name, table) in [
        ("raw_events", "raw_events"),
        ("semantic_events", "semantic_events"),
        ("embeddings", "semantic_event_embeddings"),
        ("queue_tasks", "processing_queue"),
    ] {
        let count: i64 = connection.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })?;
        counts.insert(name.to_owned(), count);
    }
    Ok(counts)
}

pub(crate) fn recent_events_on(
    connection: &Connection,
    limit: u32,
    _query: Option<&str>,
) -> Result<Vec<RawEvent>> {
    let mut statement = connection.prepare("SELECT id, timestamp_ns, event_type, source, app_name, executable_path, process_id, window_handle, window_title, element_name, text, file_path, metadata_json, privacy_class, confidence, created_at FROM raw_events ORDER BY timestamp_ns DESC LIMIT ?1")?;
    let rows = statement.query_map(params![limit], map_event)?;
    rows.collect()
}

pub(crate) fn recent_semantic_events_on(
    connection: &Connection,
    limit: u32,
    query: Option<&str>,
) -> Result<Vec<SemanticEventView>> {
    let pattern = query.map(|value| value.replace('"', ""));
    let mut statement = connection.prepare(
        "SELECT s.id, s.raw_event_id, r.timestamp_ns, r.app_name, r.window_title, s.category, s.summary, s.confidence, s.model_name, s.created_at
         FROM semantic_events s JOIN raw_events r ON r.id = s.raw_event_id
         WHERE (?1 IS NULL OR s.rowid IN (SELECT rowid FROM semantic_events_fts WHERE semantic_events_fts MATCH ?1))
         ORDER BY s.created_at DESC LIMIT ?2")?;
    let rows = statement.query_map(params![pattern, limit], |row| {
        Ok(SemanticEventView {
            id: row.get(0)?,
            raw_event_id: row.get(1)?,
            timestamp_ns: row.get(2)?,
            app_name: row.get(3)?,
            window_title: row.get(4)?,
            category: row.get(5)?,
            summary: row.get(6)?,
            confidence: row.get(7)?,
            model_name: row.get(8)?,
            created_at: row.get(9)?,
        })
    })?;
    rows.collect()
}

pub(crate) fn processing_status_for_raw_event_on(
    connection: &Connection,
    raw_event_id: &str,
) -> Result<Vec<(String, String, u32, Option<String>)>> {
    let mut statement = connection.prepare("SELECT task_type, status, attempts, error FROM processing_queue WHERE raw_event_id = ?1 ORDER BY created_at ASC")?;
    let rows = statement.query_map([raw_event_id], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
    })?;
    rows.collect()
}

pub(crate) fn queue_counts_on(connection: &Connection) -> Result<HashMap<String, i64>> {
    let mut counts = HashMap::new();
    let mut statement =
        connection.prepare("SELECT status, COUNT(*) FROM processing_queue GROUP BY status")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (status, count) = row?;
        counts.insert(status, count);
    }
    Ok(counts)
}

pub(crate) fn semantic_for_raw_event_on(
    connection: &Connection,
    raw_event_id: &str,
) -> Result<Option<SemanticEvent>> {
    connection.query_row("SELECT id, raw_event_id, category, summary, entities_json, relationships_json, confidence, model_name, model_version, created_at FROM semantic_events WHERE raw_event_id = ?1 ORDER BY created_at DESC LIMIT 1", [raw_event_id], |row| Ok(SemanticEvent { id: row.get(0)?, raw_event_id: row.get(1)?, category: row.get(2)?, summary: row.get(3)?, entities_json: row.get(4)?, relationships_json: row.get(5)?, confidence: row.get(6)?, model_name: row.get(7)?, model_version: row.get(8)?, created_at: row.get(9)? })).optional()
}

pub(crate) fn embedding_exists_on(connection: &Connection, semantic_event_id: &str) -> Result<bool> {
    connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM semantic_event_embeddings WHERE semantic_event_id = ?1)",
        [semantic_event_id],
        |row| row.get(0),
    )
}

fn map_event(row: &rusqlite::Row<'_>) -> Result<RawEvent> {
    Ok(RawEvent {
        id: row.get(0)?,
        timestamp_ns: row.get(1)?,
        event_type: row.get(2)?,
        source: row.get(3)?,
        app_name: row.get(4)?,
        executable_path: row.get(5)?,
        process_id: row.get(6)?,
        window_handle: row.get::<_, Option<i64>>(7)?.map(|value| value as u64),
        window_title: row.get(8)?,
        element_name: row.get(9)?,
        text: row.get(10)?,
        file_path: row.get(11)?,
        metadata_json: row.get(12)?,
        privacy_class: row.get(13)?,
        confidence: row.get(14)?,
        created_at: row.get(15)?,
    })
}

#[allow(dead_code)]
fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}

/// sqlite-vec accepts vectors as a JSON array literal (`[0.1, 0.2, ...]`)
/// for both inserts and `MATCH` queries against a `float[N]` column.
fn vec_literal(values: &[f32]) -> String {
    serde_json::to_string(values).unwrap_or_else(|_| "[]".into())
}

fn encode_embedding(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().ok()?));
    }
    Some(values)
}

fn add_column_if_missing(
    connection: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<()> {
    let statement = format!("ALTER TABLE {table} ADD COLUMN {column} {definition}");
    match connection.execute(&statement, []) {
        Ok(_) => Ok(()),
        Err(error) if error.to_string().contains("duplicate column name") => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[path = "tests/local_sqlite_event_database_tests.rs"]
mod tests;
