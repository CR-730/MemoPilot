CREATE TABLE session_identities (
    channel TEXT NOT NULL,
    identity_kind TEXT NOT NULL CHECK (
        identity_kind IN ('open_id', 'user_id', 'union_id')
    ),
    identity_value TEXT NOT NULL,
    session_key TEXT NOT NULL REFERENCES sessions(session_key) ON DELETE CASCADE,
    chat_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(channel, identity_kind, identity_value)
);
CREATE INDEX ix_session_identities_session
    ON session_identities(session_key);

CREATE TABLE session_interrupts (
    event_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL UNIQUE,
    session_key TEXT NOT NULL REFERENCES sessions(session_key) ON DELETE CASCADE,
    activity_version INTEGER NOT NULL CHECK (activity_version >= 0),
    requested_at TEXT NOT NULL
);

ALTER TABLE outbound_effects
    ADD COLUMN channel TEXT NOT NULL DEFAULT 'feishu';
ALTER TABLE outbound_effects
    ADD COLUMN chat_id TEXT NOT NULL DEFAULT '';
ALTER TABLE outbound_effects
    ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE outbound_effects
    ADD COLUMN last_attempt_at TEXT;
