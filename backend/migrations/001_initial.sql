CREATE TABLE IF NOT EXISTS raw_events (
    id TEXT PRIMARY KEY,
    timestamp_ns INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    app_name TEXT,
    executable_path TEXT,
    process_id INTEGER,
    window_handle INTEGER,
    window_title TEXT,
    window_bounds_json TEXT,
    automation_id TEXT,
    control_type TEXT,
    element_name TEXT,
    element_value TEXT,
    selected_text TEXT,
    key_code INTEGER,
    text TEXT,
    mouse_x REAL,
    mouse_y REAL,
    mouse_button TEXT,
    file_path TEXT,
    metadata_json TEXT NOT NULL,
    privacy_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS semantic_events (
    id TEXT PRIMARY KEY,
    raw_event_id TEXT NOT NULL REFERENCES raw_events(id),
    category TEXT NOT NULL,
    summary TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    relationships_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS semantic_events_fts USING fts5(
    category, summary, entities_json, relationships_json,
    content='semantic_events', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS semantic_events_ai AFTER INSERT ON semantic_events BEGIN
    INSERT INTO semantic_events_fts(rowid, category, summary, entities_json, relationships_json)
    VALUES (new.rowid, new.category, new.summary, new.entities_json, new.relationships_json);
END;

CREATE TRIGGER IF NOT EXISTS semantic_events_ad AFTER DELETE ON semantic_events BEGIN
    INSERT INTO semantic_events_fts(semantic_events_fts, rowid, category, summary, entities_json, relationships_json)
    VALUES ('delete', old.rowid, old.category, old.summary, old.entities_json, old.relationships_json);
END;

CREATE TRIGGER IF NOT EXISTS semantic_events_au AFTER UPDATE ON semantic_events BEGIN
    INSERT INTO semantic_events_fts(semantic_events_fts, rowid, category, summary, entities_json, relationships_json)
    VALUES ('delete', old.rowid, old.category, old.summary, old.entities_json, old.relationships_json);
    INSERT INTO semantic_events_fts(rowid, category, summary, entities_json, relationships_json)
    VALUES (new.rowid, new.category, new.summary, new.entities_json, new.relationships_json);
END;

CREATE TABLE IF NOT EXISTS processing_queue (
    id TEXT PRIMARY KEY,
    raw_event_id TEXT NOT NULL REFERENCES raw_events(id),
    task_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    model_name TEXT,
    model_version TEXT,
    error TEXT,
    retry_at TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS processing_queue_pending_idx ON processing_queue(status, priority DESC, created_at ASC);

CREATE TABLE IF NOT EXISTS semantic_event_embeddings (
    semantic_event_id TEXT PRIMARY KEY REFERENCES semantic_events(id),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding_json TEXT NOT NULL,
    embedding_blob BLOB,
    created_at TEXT NOT NULL
);

-- Hot-path lookups: newest-first raw event listing, per-raw-event semantic
-- lookup (checked on every insert and every processing task), and per-event
-- queue-status lookups from the UI and the dedupe check in
-- enqueue_unprocessed_events. Without these, each grows linearly with table
-- size and the effect compounds under capture load.
CREATE INDEX IF NOT EXISTS raw_events_timestamp_idx ON raw_events(timestamp_ns DESC);
CREATE INDEX IF NOT EXISTS semantic_events_raw_event_idx ON semantic_events(raw_event_id);
CREATE INDEX IF NOT EXISTS processing_queue_raw_event_idx ON processing_queue(raw_event_id);
