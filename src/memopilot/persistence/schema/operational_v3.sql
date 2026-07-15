ALTER TABLE outbound_effects
    ADD COLUMN cancel_on_activity INTEGER NOT NULL DEFAULT 1
    CHECK (cancel_on_activity IN (0, 1));
