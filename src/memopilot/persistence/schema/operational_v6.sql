CREATE TABLE outbound_effects_v6 (
    operation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    session_key TEXT NOT NULL REFERENCES sessions(session_key),
    payload_hash TEXT NOT NULL,
    provider_uuid TEXT NOT NULL UNIQUE,
    expected_activity_version INTEGER NOT NULL CHECK (expected_activity_version >= 0),
    state TEXT NOT NULL CHECK (
        state IN ('pending', 'sending', 'confirmed', 'cancelled', 'unknown',
                  'needs_review')
    ),
    owner_id TEXT NOT NULL,
    fencing_epoch INTEGER NOT NULL CHECK (fencing_epoch > 0),
    message_id TEXT,
    first_requested_at TEXT,
    confirmed_at TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'feishu',
    chat_id TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    last_attempt_at TEXT,
    cancel_on_activity INTEGER NOT NULL DEFAULT 1 CHECK (cancel_on_activity IN (0, 1))
);

INSERT INTO outbound_effects_v6(
    operation_id, task_id, session_key, payload_hash, provider_uuid,
    expected_activity_version, state, owner_id, fencing_epoch, message_id,
    first_requested_at, confirmed_at, error_json, created_at, updated_at,
    channel, chat_id, payload_json, last_attempt_at, cancel_on_activity
)
SELECT
    operation_id, run_id, session_key, payload_hash, provider_uuid,
    expected_activity_version, state, owner_id, fencing_epoch, message_id,
    first_requested_at, confirmed_at, error_json, created_at, updated_at,
    channel, chat_id, payload_json, last_attempt_at, cancel_on_activity
FROM outbound_effects;

DROP TABLE outbound_effects;
ALTER TABLE outbound_effects_v6 RENAME TO outbound_effects;
CREATE INDEX ix_outbound_effects_session_state
    ON outbound_effects(session_key, state, created_at);
