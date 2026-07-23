CREATE TABLE proactive_observations (
    observation_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('alert', 'context', 'content')),
    subject_id TEXT NOT NULL,
    trigger_json TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    llm_input_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX ix_proactive_observations_session
    ON proactive_observations(session_key, created_at);

ALTER TABLE hazard_snapshots
    ADD COLUMN trace_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE drift_history
    ADD COLUMN trace_json TEXT NOT NULL DEFAULT '{}';
