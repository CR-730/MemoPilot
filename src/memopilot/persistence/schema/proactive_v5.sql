ALTER TABLE pending_acknowledgements
    ADD COLUMN ttl_hours INTEGER NOT NULL DEFAULT 24 CHECK (ttl_hours > 0);

DROP TABLE content_scores;
