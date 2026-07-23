CREATE TABLE source_event_history (
    archive_id TEXT PRIMARY KEY,
    reservoir_id TEXT NOT NULL UNIQUE,
    source_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('alert', 'context', 'content')),
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    consumed_at TEXT NOT NULL,
    consume_reason TEXT,
    ack_operation_id TEXT,
    ack_state TEXT,
    ack_ttl_hours INTEGER,
    acknowledged_at TEXT,
    archived_at TEXT NOT NULL
);

CREATE INDEX ix_source_event_history_identity
    ON source_event_history(source_id, source_event_id, archived_at);
