CREATE TABLE proactive_deliveries (
    session_key TEXT NOT NULL,
    delivery_key TEXT NOT NULL,
    message TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    PRIMARY KEY(session_key, delivery_key)
);

CREATE INDEX ix_proactive_deliveries_sent_at
    ON proactive_deliveries(sent_at);
