CREATE TABLE memory_sources (
    source_ref TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES memory_items(item_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);
CREATE INDEX ix_memory_sources_item ON memory_sources(item_id);

INSERT OR IGNORE INTO memory_sources(source_ref, item_id, created_at)
SELECT source_ref, item_id, created_at FROM memory_items;

CREATE TABLE memory_vector_rows (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL UNIQUE REFERENCES memory_items(item_id) ON DELETE CASCADE
);

CREATE TABLE memory_ingestion_batches (
    batch_id TEXT PRIMARY KEY,
    source_ref TEXT NOT NULL UNIQUE,
    model_output_json TEXT,
    state TEXT NOT NULL CHECK (state IN ('pending', 'writing', 'committed', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    committed_at TEXT
);

CREATE VIRTUAL TABLE memory_fts USING fts5(
    item_id UNINDEXED,
    terms,
    tokenize='unicode61'
);
