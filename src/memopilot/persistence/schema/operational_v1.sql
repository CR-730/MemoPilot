CREATE TABLE sessions (
    session_key TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE session_activity (
    session_key TEXT PRIMARY KEY REFERENCES sessions(session_key) ON DELETE CASCADE,
    activity_version INTEGER NOT NULL DEFAULT 0 CHECK (activity_version >= 0),
    last_user_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE inbound_events (
    event_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL UNIQUE,
    session_key TEXT NOT NULL REFERENCES sessions(session_key),
    payload_json TEXT NOT NULL,
    received_at TEXT NOT NULL
);
CREATE INDEX ix_inbound_events_session_received
    ON inbound_events(session_key, received_at);

CREATE TABLE agent_jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    priority INTEGER NOT NULL CHECK (priority BETWEEN 0 AND 3),
    session_key TEXT NOT NULL REFERENCES sessions(session_key),
    idempotency_key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (
        state IN ('queued', 'running', 'succeeded', 'skipped', 'failed',
                  'cancelled', 'needs_review')
    ),
    activity_version INTEGER NOT NULL CHECK (activity_version >= 0),
    payload_json TEXT NOT NULL DEFAULT '{}',
    heartbeat_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX ix_agent_jobs_state_priority ON agent_jobs(state, priority, created_at);
CREATE INDEX ix_agent_jobs_session_state ON agent_jobs(session_key, state);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL UNIQUE REFERENCES agent_jobs(job_id),
    owner_id TEXT,
    fencing_epoch INTEGER CHECK (fencing_epoch IS NULL OR fencing_epoch > 0),
    state TEXT NOT NULL CHECK (
        state IN ('queued', 'running', 'recovering', 'succeeded', 'skipped',
                  'failed', 'cancelled', 'needs_review')
    ),
    started_at TEXT,
    heartbeat_at TEXT,
    finished_at TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE run_attempts (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
    owner_id TEXT NOT NULL,
    fencing_epoch INTEGER NOT NULL CHECK (fencing_epoch > 0),
    outcome TEXT CHECK (
        outcome IS NULL OR outcome IN ('succeeded', 'failed', 'lost_lease',
                                       'cancelled', 'needs_review')
    ),
    started_at TEXT NOT NULL,
    heartbeat_at TEXT,
    finished_at TEXT,
    error_json TEXT,
    UNIQUE(run_id, attempt_no)
);

CREATE TABLE steps (
    step_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL CHECK (step_index >= 0),
    phase TEXT NOT NULL,
    step_type TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('pending', 'running', 'succeeded', 'failed', 'skipped',
                  'cancelled', 'unknown')
    ),
    tool_name TEXT,
    input_json TEXT,
    observation_json TEXT,
    owner_id TEXT,
    fencing_epoch INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(run_id, step_index)
);

CREATE TABLE session_fences (
    session_key TEXT PRIMARY KEY REFERENCES sessions(session_key) ON DELETE CASCADE,
    current_epoch INTEGER NOT NULL DEFAULT 0 CHECK (current_epoch >= 0),
    owner_id TEXT,
    heartbeat_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE outbox_events (
    outbox_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('pending', 'publishing', 'published', 'dead')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    recovery_count INTEGER NOT NULL DEFAULT 0 CHECK (recovery_count >= 0),
    next_attempt_at TEXT NOT NULL,
    claim_owner TEXT,
    claim_until TEXT,
    published_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX ix_outbox_events_dispatch
    ON outbox_events(state, next_attempt_at, claim_until);

CREATE TABLE outbound_effects (
    operation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    session_key TEXT NOT NULL REFERENCES sessions(session_key),
    payload_hash TEXT NOT NULL,
    provider_uuid TEXT NOT NULL UNIQUE,
    expected_activity_version INTEGER NOT NULL CHECK (expected_activity_version >= 0),
    state TEXT NOT NULL CHECK (
        state IN ('pending', 'sending', 'confirmed', 'cancelled', 'unknown',
                  'needs_review')
    ),
    owner_id TEXT NOT NULL,
    fencing_epoch INTEGER NOT NULL CHECK (fencing_epoch > 0),
    message_id TEXT,
    first_requested_at TEXT,
    confirmed_at TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX ix_outbound_effects_session_state
    ON outbound_effects(session_key, state, created_at);

CREATE TABLE scheduled_tasks (
    task_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL REFERENCES sessions(session_key),
    schedule_kind TEXT NOT NULL CHECK (schedule_kind IN ('at', 'after', 'every')),
    schedule_json TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (execution_mode IN ('instant', 'agent')),
    payload_json TEXT NOT NULL,
    next_run_at TEXT,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX ix_scheduled_tasks_due ON scheduled_tasks(enabled, next_run_at);

CREATE TABLE scheduled_executions (
    execution_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES scheduled_tasks(task_id),
    scheduled_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('queued', 'running', 'succeeded', 'skipped', 'failed',
                  'cancelled', 'needs_review')
    ),
    job_id TEXT UNIQUE REFERENCES agent_jobs(job_id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(task_id, scheduled_at)
);

CREATE TABLE consolidation_manifests (
    consolidation_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL REFERENCES sessions(session_key),
    first_message_id TEXT NOT NULL,
    last_message_id TEXT NOT NULL,
    artifact_hashes_json TEXT NOT NULL,
    model_output_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'writing', 'committed', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    committed_at TEXT
);

CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL REFERENCES sessions(session_key),
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    turn_position INTEGER NOT NULL CHECK (turn_position >= 0),
    owner_id TEXT,
    fencing_epoch INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(turn_id, turn_position)
);
CREATE INDEX ix_messages_session_created ON messages(session_key, created_at);
