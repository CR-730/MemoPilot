ALTER TABLE messages
    ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';
