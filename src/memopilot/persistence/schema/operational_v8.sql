CREATE TABLE scheduled_executions_v8 (
    execution_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES scheduled_tasks(task_id),
    scheduled_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('queued', 'running', 'succeeded', 'skipped', 'failed', 'cancelled')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(task_id, scheduled_at)
);

INSERT INTO scheduled_executions_v8(
    execution_id, task_id, scheduled_at, state, created_at, updated_at
)
SELECT execution_id, task_id, scheduled_at,
       CASE WHEN state = 'needs_review' THEN 'failed' ELSE state END,
       created_at, updated_at
FROM scheduled_executions;

DROP TABLE scheduled_executions;
ALTER TABLE scheduled_executions_v8 RENAME TO scheduled_executions;
DROP TABLE outbound_effects;
