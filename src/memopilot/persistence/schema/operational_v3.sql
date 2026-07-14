ALTER TABLE sessions
    ADD COLUMN last_consolidated_position INTEGER NOT NULL DEFAULT 0
    CHECK (last_consolidated_position >= 0);

ALTER TABLE messages
    ADD COLUMN session_position INTEGER;

CREATE UNIQUE INDEX ux_messages_session_position
    ON messages(session_key, session_position)
    WHERE session_position IS NOT NULL;

CREATE INDEX ix_messages_session_position
    ON messages(session_key, session_position);

ALTER TABLE consolidation_manifests
    ADD COLUMN artifact_states_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE consolidation_manifests
    ADD COLUMN last_error TEXT;
