DROP INDEX ix_proactive_decisions_pending_session;
DROP INDEX ix_proactive_decisions_session;

CREATE TABLE proactive_decisions_v9 (
    decision_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    trigger_kind TEXT NOT NULL CHECK (trigger_kind IN ('alert', 'context', 'content')),
    action TEXT NOT NULL CHECK (action IN ('share', 'skip', 'alert', 'send_event', 'skip_event')),
    source_events_json TEXT NOT NULL,
    activity_version INTEGER NOT NULL CHECK (activity_version >= 0),
    state TEXT NOT NULL CHECK (state IN ('pending', 'committed', 'failed', 'cancelled')),
    reason TEXT,
    decided_at TEXT NOT NULL,
    committed_at TEXT,
    decision_payload_json TEXT NOT NULL DEFAULT '{}'
);

INSERT INTO proactive_decisions_v9(
    decision_id, session_key, trigger_kind, action, source_events_json,
    activity_version, state, reason, decided_at, committed_at, decision_payload_json
)
SELECT decision_id, session_key, trigger_kind, action, source_events_json,
       activity_version, state, reason, decided_at, committed_at, decision_payload_json
FROM proactive_decisions;

DROP TABLE proactive_decisions;
ALTER TABLE proactive_decisions_v9 RENAME TO proactive_decisions;

CREATE INDEX ix_proactive_decisions_session
    ON proactive_decisions(session_key, decided_at);
CREATE INDEX ix_proactive_decisions_pending_session
    ON proactive_decisions(session_key, state, decided_at);
