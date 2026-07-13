CREATE TABLE source_events (
    reservoir_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('alert', 'context', 'content')),
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    consumed_at TEXT,
    consume_reason TEXT,
    UNIQUE(source_id, source_event_id)
);
CREATE INDEX ix_source_events_pending
    ON source_events(session_key, kind, consumed_at, occurred_at);

CREATE TABLE source_cursors (
    source_id TEXT PRIMARY KEY,
    cursor_value TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE pending_acknowledgements (
    acknowledgement_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    ack_operation_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('pending', 'acknowledged', 'dead')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at TEXT NOT NULL,
    acknowledged_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_id, source_event_id)
        REFERENCES source_events(source_id, source_event_id)
);
CREATE INDEX ix_pending_ack_due
    ON pending_acknowledgements(state, next_attempt_at);

CREATE TABLE content_scores (
    source_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    source_score REAL NOT NULL,
    semantic_score REAL NOT NULL,
    freshness_score REAL NOT NULL,
    event_score REAL NOT NULL,
    weights_json TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY(source_id, source_event_id),
    FOREIGN KEY(source_id, source_event_id)
        REFERENCES source_events(source_id, source_event_id) ON DELETE CASCADE
);

CREATE TABLE hazard_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    previous_hazard REAL NOT NULL CHECK (previous_hazard >= 0),
    current_hazard REAL NOT NULL CHECK (current_hazard >= 0),
    threshold REAL NOT NULL CHECK (threshold >= 0),
    elapsed_seconds REAL NOT NULL CHECK (elapsed_seconds >= 0),
    top_event_ids_json TEXT NOT NULL,
    calculated_at TEXT NOT NULL
);
CREATE INDEX ix_hazard_snapshots_session
    ON hazard_snapshots(session_key, calculated_at);

CREATE TABLE wake_decisions (
    decision_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    trigger_kind TEXT NOT NULL CHECK (trigger_kind IN ('alert', 'context', 'content')),
    action TEXT NOT NULL CHECK (action IN ('share', 'skip', 'alert', 'send_event', 'skip_event')),
    source_events_json TEXT NOT NULL,
    effect_operation_id TEXT UNIQUE,
    activity_version INTEGER NOT NULL CHECK (activity_version >= 0),
    state TEXT NOT NULL CHECK (state IN ('pending', 'committed', 'failed', 'cancelled')),
    reason TEXT,
    decided_at TEXT NOT NULL,
    committed_at TEXT
);
CREATE INDEX ix_wake_decisions_session ON wake_decisions(session_key, decided_at);

CREATE TABLE drift_history (
    drift_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    skill_name TEXT,
    drive_before REAL NOT NULL CHECK (drive_before >= 0),
    threshold REAL NOT NULL CHECK (threshold >= 0),
    outcome TEXT NOT NULL CHECK (
        outcome IN ('suppressed', 'queued', 'running', 'succeeded', 'failed', 'cancelled')
    ),
    job_id TEXT,
    reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX ix_drift_history_session ON drift_history(session_key, created_at);
