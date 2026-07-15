ALTER TABLE session_interrupts
    ADD COLUMN target_run_id TEXT REFERENCES runs(run_id);
ALTER TABLE session_interrupts
    ADD COLUMN state TEXT NOT NULL DEFAULT 'pending'
    CHECK (state IN ('pending', 'acknowledged', 'no_active'));
ALTER TABLE session_interrupts
    ADD COLUMN acknowledged_at TEXT;

CREATE INDEX ix_session_interrupts_target_state
    ON session_interrupts(target_run_id, state);

CREATE TABLE turn_interrupt_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id) ON DELETE CASCADE,
    session_key TEXT NOT NULL REFERENCES sessions(session_key) ON DELETE CASCADE,
    original_message TEXT NOT NULL,
    partial_reply TEXT NOT NULL DEFAULT '',
    partial_thinking TEXT NOT NULL DEFAULT '',
    tools_json TEXT NOT NULL DEFAULT '[]',
    tool_chain_json TEXT NOT NULL DEFAULT '[]',
    interrupted_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    reserved_job_id TEXT,
    consumed_at TEXT
);
CREATE INDEX ix_turn_interrupt_snapshots_resumable
    ON turn_interrupt_snapshots(session_key, consumed_at, expires_at, interrupted_at);
