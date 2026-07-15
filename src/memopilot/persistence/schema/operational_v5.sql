ALTER TABLE sessions
    ADD COLUMN last_consolidated_position INTEGER NOT NULL DEFAULT 0
    CHECK (last_consolidated_position >= 0);

ALTER TABLE messages
    ADD COLUMN session_position INTEGER;

WITH ranked_messages AS (
    SELECT
        message_id,
        ROW_NUMBER() OVER (
            PARTITION BY session_key
            ORDER BY created_at, turn_id, turn_position, message_id
        ) AS position
    FROM messages
)
UPDATE messages
SET session_position = (
    SELECT position
    FROM ranked_messages
    WHERE ranked_messages.message_id = messages.message_id
)
WHERE session_position IS NULL;

CREATE UNIQUE INDEX ux_messages_session_position
    ON messages(session_key, session_position)
    WHERE session_position IS NOT NULL;

CREATE INDEX ix_messages_session_position
    ON messages(session_key, session_position);

ALTER TABLE consolidation_manifests
    ADD COLUMN artifact_states_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE consolidation_manifests
    ADD COLUMN last_error TEXT;
