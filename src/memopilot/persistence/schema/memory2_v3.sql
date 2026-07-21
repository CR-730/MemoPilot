CREATE TABLE memory_usages (
    usage_ref TEXT NOT NULL,
    item_id TEXT NOT NULL REFERENCES memory_items(item_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (usage_ref, item_id)
);
CREATE INDEX ix_memory_usages_item ON memory_usages(item_id);
