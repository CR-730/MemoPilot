CREATE TABLE hazard_state (
    session_key TEXT PRIMARY KEY,
    hazard REAL NOT NULL CHECK (hazard >= 0),
    threshold REAL NOT NULL CHECK (threshold >= 0),
    updated_at TEXT NOT NULL,
    last_proactive_at TEXT
);

CREATE TABLE context_states (
    source_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE context_reevaluation_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    last_signaled_at TEXT,
    last_candidate_at TEXT,
    suppressed_count INTEGER NOT NULL DEFAULT 0 CHECK (suppressed_count >= 0)
);

CREATE TABLE drift_state (
    session_key TEXT PRIMARY KEY,
    drive REAL NOT NULL CHECK (drive >= 0),
    threshold REAL NOT NULL CHECK (threshold >= 0),
    updated_at TEXT NOT NULL,
    last_drift_at TEXT,
    fingerprint TEXT NOT NULL DEFAULT '',
    repeat_count INTEGER NOT NULL DEFAULT 0 CHECK (repeat_count >= 0)
);
