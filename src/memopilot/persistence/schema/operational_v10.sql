ALTER TABLE messages
    ADD COLUMN tool_chain_json TEXT NOT NULL DEFAULT '[]';
