CREATE TABLE memory_items (
    item_id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL CHECK (
        memory_type IN ('event', 'profile', 'preference', 'procedure')
    ),
    summary TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_ref TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded')),
    reinforcement_count INTEGER NOT NULL DEFAULT 1 CHECK (reinforcement_count > 0),
    emotional_weight INTEGER NOT NULL DEFAULT 0 CHECK (emotional_weight BETWEEN 0 AND 10),
    happened_at TEXT,
    extra_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(memory_type, content_hash)
);
CREATE INDEX ix_memory_items_type_status ON memory_items(memory_type, status, updated_at);

CREATE TABLE memory_embeddings (
    item_id TEXT PRIMARY KEY REFERENCES memory_items(item_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    embedding BLOB NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX ix_memory_embeddings_model ON memory_embeddings(provider, model, dimension);

CREATE TABLE memory_metadata (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE keyword_index_metadata (
    index_name TEXT PRIMARY KEY,
    tokenizer TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    last_rebuilt_at TEXT,
    item_count INTEGER NOT NULL DEFAULT 0 CHECK (item_count >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE memory_relations (
    relation_id TEXT PRIMARY KEY,
    source_item_id TEXT NOT NULL REFERENCES memory_items(item_id),
    target_item_id TEXT NOT NULL REFERENCES memory_items(item_id),
    relation_type TEXT NOT NULL CHECK (relation_type IN ('reinforce', 'supersede')),
    source_ref TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    CHECK (source_item_id <> target_item_id),
    UNIQUE(source_item_id, target_item_id, relation_type)
);
CREATE INDEX ix_memory_relations_target ON memory_relations(target_item_id, relation_type);
