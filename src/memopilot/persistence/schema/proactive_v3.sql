ALTER TABLE proactive_decisions
    ADD COLUMN decision_payload_json TEXT NOT NULL DEFAULT '{}';

CREATE INDEX ix_proactive_decisions_pending_session
    ON proactive_decisions(session_key, state, decided_at);
