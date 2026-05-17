from __future__ import annotations


class BatchPhase:
    PLANNED = "planned"
    ATTEMPTING = "attempting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RECOVERY_DESIRED = "recovery_desired"
    RECOVERING = "recovering"
    RECOVERY_SUCCEEDED = "recovery_succeeded"
    RECOVERY_FAILED = "recovery_failed"


_VALID_TRANSITIONS = {
    BatchPhase.PLANNED: frozenset({BatchPhase.ATTEMPTING}),
    BatchPhase.ATTEMPTING: frozenset({BatchPhase.SUCCEEDED, BatchPhase.FAILED}),
    BatchPhase.SUCCEEDED: frozenset[str](),
    BatchPhase.FAILED: frozenset({BatchPhase.RECOVERY_DESIRED}),
    BatchPhase.RECOVERY_DESIRED: frozenset({BatchPhase.RECOVERING}),
    BatchPhase.RECOVERING: frozenset({BatchPhase.RECOVERY_SUCCEEDED, BatchPhase.RECOVERY_FAILED}),
    BatchPhase.RECOVERY_SUCCEEDED: frozenset[str](),
    BatchPhase.RECOVERY_FAILED: frozenset({BatchPhase.RECOVERY_DESIRED}),
}


_BATCH_TABLE = "operation_batches"
_OPERATION_RUN_TABLE = "operation_runs"
_OPERATION_PHASE_TABLE = "operation_phase_records"
_OPERATION_TABLE = "operation_journal_operations"
_CHECKPOINT_TABLE = "operation_journal_checkpoints"
_RECOVERY_TABLE = "operation_recovery_records"
_RECOVERY_ACTION_TABLE = "operation_recovery_action_records"
_ARTIFACT_CLEANUP_TABLE = "operation_artifact_cleanup_records"


class _BatchColumn:
    STORAGE_ORDER = "storage_order"
    BATCH_ID = "batch_id"
    IDEMPOTENCY_KEY = "idempotency_key"
    LEASE_NAME = "lease_name"
    LEASE_OWNER = "lease_owner"
    LEASE_FENCING_TOKEN = "lease_fencing_token"
    LEASE_TOKEN_DIGEST = "lease_token_digest"
    OWNER = "owner"
    RUN_ID = "run_id"
    OPERATION_RUN_ID = "operation_run_id"
    OPERATION_PHASE_ID = "operation_phase_id"
    RESOURCE_KEY = "resource_key"
    CLAIM_OWNER = "claim_owner"
    CLAIM_SCOPE = "claim_scope"
    PHASE = "phase"
    PAYLOAD = "payload"
    STATUS_MESSAGE = "status_message"
    STATUS_PAYLOAD = "status_payload"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class _OperationColumn:
    OPERATION_ID = "operation_id"
    BATCH_ID = "batch_id"
    SEQUENCE = "sequence"
    OPERATION_TYPE = "operation_type"
    RESOURCE_KEY = "resource_key"
    PAYLOAD = "payload"
    CREATED_AT = "created_at"


class _OperationRunColumn:
    OPERATION_RUN_ID = "operation_run_id"
    RUN_ID = "run_id"
    OWNER = "owner"
    LEASE_NAME = "lease_name"
    LEASE_FENCING_TOKEN = "lease_fencing_token"
    STATUS = "status"
    PAYLOAD = "payload"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class _OperationPhaseColumn:
    OPERATION_PHASE_ID = "operation_phase_id"
    OPERATION_RUN_ID = "operation_run_id"
    PHASE_NAME = "phase_name"
    STATUS = "status"
    PHASE_ORDER = "phase_order"
    PAYLOAD = "payload"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class _CheckpointColumn:
    CHECKPOINT_ID = "checkpoint_id"
    BATCH_ID = "batch_id"
    SEQUENCE = "sequence"
    OPERATION_ID = "operation_id"
    RESOURCE_KEY = "resource_key"
    CHECKPOINT_TYPE = "checkpoint_type"
    PAYLOAD = "payload"
    CREATED_AT = "created_at"


class _RecoveryColumn:
    RECOVERY_ID = "recovery_id"
    BATCH_ID = "batch_id"
    SEQUENCE = "sequence"
    PHASE = "phase"
    REASON = "reason"
    PAYLOAD = "payload"
    CREATED_AT = "created_at"


class _RecoveryActionColumn:
    ACTION_RECORD_ID = "action_record_id"
    ACTION_ID = "action_id"
    RECOVERY_ATTEMPT_ID = "recovery_attempt_id"
    BATCH_ID = "batch_id"
    SEQUENCE = "sequence"
    ACTION_TYPE = "action_type"
    STATUS = "status"
    RESOURCE_KEY = "resource_key"
    REASON = "reason"
    PAYLOAD = "payload"
    CREATED_AT = "created_at"


class _ArtifactCleanupColumn:
    CLEANUP_RECORD_ID = "cleanup_record_id"
    ARTIFACT_ID = "artifact_id"
    BATCH_ID = "batch_id"
    SEQUENCE = "sequence"
    TRIGGER = "trigger"
    STATUS = "status"
    RESOURCE_KEY = "resource_key"
    REASON = "reason"
    PAYLOAD = "payload"
    CREATED_AT = "created_at"


_BATCH_SELECT_COLUMNS = (
    f"rowid AS {_BatchColumn.STORAGE_ORDER}",
    _BatchColumn.BATCH_ID,
    _BatchColumn.IDEMPOTENCY_KEY,
    _BatchColumn.LEASE_NAME,
    _BatchColumn.LEASE_OWNER,
    _BatchColumn.LEASE_FENCING_TOKEN,
    _BatchColumn.LEASE_TOKEN_DIGEST,
    _BatchColumn.OWNER,
    _BatchColumn.RUN_ID,
    _BatchColumn.OPERATION_RUN_ID,
    _BatchColumn.OPERATION_PHASE_ID,
    _BatchColumn.RESOURCE_KEY,
    _BatchColumn.CLAIM_OWNER,
    _BatchColumn.CLAIM_SCOPE,
    _BatchColumn.PHASE,
    _BatchColumn.PAYLOAD,
    _BatchColumn.STATUS_MESSAGE,
    _BatchColumn.STATUS_PAYLOAD,
    _BatchColumn.CREATED_AT,
    _BatchColumn.UPDATED_AT,
)
_BATCH_SELECT_LIST = ", ".join(_BATCH_SELECT_COLUMNS)

_OPERATION_RUN_SELECT_COLUMNS = (
    _OperationRunColumn.OPERATION_RUN_ID,
    _OperationRunColumn.RUN_ID,
    _OperationRunColumn.OWNER,
    _OperationRunColumn.LEASE_NAME,
    _OperationRunColumn.LEASE_FENCING_TOKEN,
    _OperationRunColumn.STATUS,
    _OperationRunColumn.PAYLOAD,
    _OperationRunColumn.CREATED_AT,
    _OperationRunColumn.UPDATED_AT,
)
_OPERATION_RUN_SELECT_LIST = ", ".join(_OPERATION_RUN_SELECT_COLUMNS)

_OPERATION_PHASE_SELECT_COLUMNS = (
    _OperationPhaseColumn.OPERATION_PHASE_ID,
    _OperationPhaseColumn.OPERATION_RUN_ID,
    _OperationPhaseColumn.PHASE_NAME,
    _OperationPhaseColumn.STATUS,
    _OperationPhaseColumn.PHASE_ORDER,
    _OperationPhaseColumn.PAYLOAD,
    _OperationPhaseColumn.CREATED_AT,
    _OperationPhaseColumn.UPDATED_AT,
)
_OPERATION_PHASE_SELECT_LIST = ", ".join(_OPERATION_PHASE_SELECT_COLUMNS)

_OPERATION_SELECT_COLUMNS = (
    _OperationColumn.OPERATION_ID,
    _OperationColumn.BATCH_ID,
    _OperationColumn.SEQUENCE,
    _OperationColumn.OPERATION_TYPE,
    _OperationColumn.RESOURCE_KEY,
    _OperationColumn.PAYLOAD,
    _OperationColumn.CREATED_AT,
)
_OPERATION_SELECT_LIST = ", ".join(_OPERATION_SELECT_COLUMNS)

_CHECKPOINT_SELECT_COLUMNS = (
    _CheckpointColumn.CHECKPOINT_ID,
    _CheckpointColumn.BATCH_ID,
    _CheckpointColumn.SEQUENCE,
    _CheckpointColumn.OPERATION_ID,
    _CheckpointColumn.RESOURCE_KEY,
    _CheckpointColumn.CHECKPOINT_TYPE,
    _CheckpointColumn.PAYLOAD,
    _CheckpointColumn.CREATED_AT,
)
_CHECKPOINT_SELECT_LIST = ", ".join(_CHECKPOINT_SELECT_COLUMNS)

_RECOVERY_SELECT_COLUMNS = (
    _RecoveryColumn.RECOVERY_ID,
    _RecoveryColumn.BATCH_ID,
    _RecoveryColumn.SEQUENCE,
    _RecoveryColumn.PHASE,
    _RecoveryColumn.REASON,
    _RecoveryColumn.PAYLOAD,
    _RecoveryColumn.CREATED_AT,
)
_RECOVERY_SELECT_LIST = ", ".join(_RECOVERY_SELECT_COLUMNS)

_RECOVERY_ACTION_SELECT_COLUMNS = (
    _RecoveryActionColumn.ACTION_RECORD_ID,
    _RecoveryActionColumn.ACTION_ID,
    _RecoveryActionColumn.RECOVERY_ATTEMPT_ID,
    _RecoveryActionColumn.BATCH_ID,
    _RecoveryActionColumn.SEQUENCE,
    _RecoveryActionColumn.ACTION_TYPE,
    _RecoveryActionColumn.STATUS,
    _RecoveryActionColumn.RESOURCE_KEY,
    _RecoveryActionColumn.REASON,
    _RecoveryActionColumn.PAYLOAD,
    _RecoveryActionColumn.CREATED_AT,
)
_RECOVERY_ACTION_SELECT_LIST = ", ".join(_RECOVERY_ACTION_SELECT_COLUMNS)

_ARTIFACT_CLEANUP_SELECT_COLUMNS = (
    _ArtifactCleanupColumn.CLEANUP_RECORD_ID,
    _ArtifactCleanupColumn.ARTIFACT_ID,
    _ArtifactCleanupColumn.BATCH_ID,
    _ArtifactCleanupColumn.SEQUENCE,
    _ArtifactCleanupColumn.TRIGGER,
    _ArtifactCleanupColumn.STATUS,
    _ArtifactCleanupColumn.RESOURCE_KEY,
    _ArtifactCleanupColumn.REASON,
    _ArtifactCleanupColumn.PAYLOAD,
    _ArtifactCleanupColumn.CREATED_AT,
)
_ARTIFACT_CLEANUP_SELECT_LIST = ", ".join(_ARTIFACT_CLEANUP_SELECT_COLUMNS)

_JOURNAL_SCHEMA = (
    f"""
    CREATE TABLE IF NOT EXISTS {_BATCH_TABLE} (
        {_BatchColumn.BATCH_ID} TEXT PRIMARY KEY,
        {_BatchColumn.IDEMPOTENCY_KEY} TEXT NOT NULL UNIQUE,
        {_BatchColumn.LEASE_NAME} TEXT NOT NULL,
        {_BatchColumn.LEASE_OWNER} TEXT,
        {_BatchColumn.LEASE_FENCING_TOKEN} INTEGER NOT NULL,
        {_BatchColumn.LEASE_TOKEN_DIGEST} TEXT,
        {_BatchColumn.OWNER} TEXT NOT NULL,
        {_BatchColumn.RUN_ID} TEXT NOT NULL,
        {_BatchColumn.OPERATION_RUN_ID} TEXT,
        {_BatchColumn.OPERATION_PHASE_ID} TEXT,
        {_BatchColumn.RESOURCE_KEY} TEXT,
        {_BatchColumn.CLAIM_OWNER} TEXT,
        {_BatchColumn.CLAIM_SCOPE} TEXT,
        {_BatchColumn.PHASE} TEXT NOT NULL,
        {_BatchColumn.PAYLOAD} TEXT NOT NULL,
        {_BatchColumn.STATUS_MESSAGE} TEXT,
        {_BatchColumn.STATUS_PAYLOAD} TEXT NOT NULL,
        {_BatchColumn.CREATED_AT} TEXT NOT NULL,
        {_BatchColumn.UPDATED_AT} TEXT NOT NULL
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_BATCH_TABLE}_run_id
    ON {_BATCH_TABLE} ({_BatchColumn.RUN_ID}, {_BatchColumn.CREATED_AT}, {_BatchColumn.BATCH_ID})
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {_OPERATION_RUN_TABLE} (
        {_OperationRunColumn.OPERATION_RUN_ID} TEXT PRIMARY KEY,
        {_OperationRunColumn.RUN_ID} TEXT NOT NULL,
        {_OperationRunColumn.OWNER} TEXT NOT NULL,
        {_OperationRunColumn.LEASE_NAME} TEXT NOT NULL,
        {_OperationRunColumn.LEASE_FENCING_TOKEN} INTEGER NOT NULL,
        {_OperationRunColumn.STATUS} TEXT NOT NULL,
        {_OperationRunColumn.PAYLOAD} TEXT NOT NULL,
        {_OperationRunColumn.CREATED_AT} TEXT NOT NULL,
        {_OperationRunColumn.UPDATED_AT} TEXT NOT NULL
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_OPERATION_RUN_TABLE}_run_owner
    ON {_OPERATION_RUN_TABLE} (
        {_OperationRunColumn.RUN_ID},
        {_OperationRunColumn.OWNER},
        {_OperationRunColumn.CREATED_AT},
        {_OperationRunColumn.OPERATION_RUN_ID}
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {_OPERATION_PHASE_TABLE} (
        {_OperationPhaseColumn.OPERATION_PHASE_ID} TEXT PRIMARY KEY,
        {_OperationPhaseColumn.OPERATION_RUN_ID} TEXT NOT NULL
            REFERENCES {_OPERATION_RUN_TABLE} ({_OperationRunColumn.OPERATION_RUN_ID}) ON DELETE CASCADE,
        {_OperationPhaseColumn.PHASE_NAME} TEXT NOT NULL,
        {_OperationPhaseColumn.STATUS} TEXT NOT NULL,
        {_OperationPhaseColumn.PHASE_ORDER} INTEGER NOT NULL,
        {_OperationPhaseColumn.PAYLOAD} TEXT NOT NULL,
        {_OperationPhaseColumn.CREATED_AT} TEXT NOT NULL,
        {_OperationPhaseColumn.UPDATED_AT} TEXT NOT NULL,
        UNIQUE ({_OperationPhaseColumn.OPERATION_RUN_ID}, {_OperationPhaseColumn.PHASE_NAME})
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_OPERATION_PHASE_TABLE}_run_order
    ON {_OPERATION_PHASE_TABLE} (
        {_OperationPhaseColumn.OPERATION_RUN_ID},
        {_OperationPhaseColumn.PHASE_ORDER},
        {_OperationPhaseColumn.OPERATION_PHASE_ID}
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {_OPERATION_TABLE} (
        {_OperationColumn.OPERATION_ID} TEXT PRIMARY KEY,
        {_OperationColumn.BATCH_ID} TEXT NOT NULL REFERENCES {_BATCH_TABLE} ({_BatchColumn.BATCH_ID}) ON DELETE CASCADE,
        {_OperationColumn.SEQUENCE} INTEGER NOT NULL,
        {_OperationColumn.OPERATION_TYPE} TEXT NOT NULL,
        {_OperationColumn.RESOURCE_KEY} TEXT,
        {_OperationColumn.PAYLOAD} TEXT NOT NULL,
        {_OperationColumn.CREATED_AT} TEXT NOT NULL,
        UNIQUE ({_OperationColumn.BATCH_ID}, {_OperationColumn.SEQUENCE})
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_OPERATION_TABLE}_batch_sequence
    ON {_OPERATION_TABLE} ({_OperationColumn.BATCH_ID}, {_OperationColumn.SEQUENCE}, {_OperationColumn.OPERATION_ID})
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {_CHECKPOINT_TABLE} (
        {_CheckpointColumn.CHECKPOINT_ID} TEXT PRIMARY KEY,
        {_CheckpointColumn.BATCH_ID} TEXT NOT NULL REFERENCES {_BATCH_TABLE} ({_BatchColumn.BATCH_ID})
            ON DELETE CASCADE,
        {_CheckpointColumn.SEQUENCE} INTEGER NOT NULL,
        {_CheckpointColumn.OPERATION_ID} TEXT REFERENCES {_OPERATION_TABLE} ({_OperationColumn.OPERATION_ID})
            ON DELETE SET NULL,
        {_CheckpointColumn.RESOURCE_KEY} TEXT NOT NULL,
        {_CheckpointColumn.CHECKPOINT_TYPE} TEXT NOT NULL,
        {_CheckpointColumn.PAYLOAD} TEXT NOT NULL,
        {_CheckpointColumn.CREATED_AT} TEXT NOT NULL,
        UNIQUE ({_CheckpointColumn.BATCH_ID}, {_CheckpointColumn.SEQUENCE})
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_CHECKPOINT_TABLE}_batch_sequence
    ON {_CHECKPOINT_TABLE} (
        {_CheckpointColumn.BATCH_ID},
        {_CheckpointColumn.SEQUENCE},
        {_CheckpointColumn.CHECKPOINT_ID}
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {_RECOVERY_TABLE} (
        {_RecoveryColumn.RECOVERY_ID} TEXT PRIMARY KEY,
        {_RecoveryColumn.BATCH_ID} TEXT NOT NULL REFERENCES {_BATCH_TABLE} ({_BatchColumn.BATCH_ID}) ON DELETE CASCADE,
        {_RecoveryColumn.SEQUENCE} INTEGER NOT NULL,
        {_RecoveryColumn.PHASE} TEXT NOT NULL,
        {_RecoveryColumn.REASON} TEXT,
        {_RecoveryColumn.PAYLOAD} TEXT NOT NULL,
        {_RecoveryColumn.CREATED_AT} TEXT NOT NULL,
        UNIQUE ({_RecoveryColumn.BATCH_ID}, {_RecoveryColumn.SEQUENCE})
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_RECOVERY_TABLE}_batch_sequence
    ON {_RECOVERY_TABLE} ({_RecoveryColumn.BATCH_ID}, {_RecoveryColumn.SEQUENCE}, {_RecoveryColumn.RECOVERY_ID})
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {_RECOVERY_ACTION_TABLE} (
        {_RecoveryActionColumn.ACTION_RECORD_ID} TEXT PRIMARY KEY,
        {_RecoveryActionColumn.ACTION_ID} TEXT NOT NULL,
        {_RecoveryActionColumn.RECOVERY_ATTEMPT_ID} TEXT NOT NULL
            REFERENCES {_RECOVERY_TABLE} ({_RecoveryColumn.RECOVERY_ID}) ON DELETE CASCADE,
        {_RecoveryActionColumn.BATCH_ID} TEXT NOT NULL REFERENCES {_BATCH_TABLE} ({_BatchColumn.BATCH_ID})
            ON DELETE CASCADE,
        {_RecoveryActionColumn.SEQUENCE} INTEGER NOT NULL,
        {_RecoveryActionColumn.ACTION_TYPE} TEXT NOT NULL,
        {_RecoveryActionColumn.STATUS} TEXT NOT NULL,
        {_RecoveryActionColumn.RESOURCE_KEY} TEXT,
        {_RecoveryActionColumn.REASON} TEXT,
        {_RecoveryActionColumn.PAYLOAD} TEXT NOT NULL,
        {_RecoveryActionColumn.CREATED_AT} TEXT NOT NULL,
        UNIQUE ({_RecoveryActionColumn.BATCH_ID}, {_RecoveryActionColumn.SEQUENCE})
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_RECOVERY_ACTION_TABLE}_batch_sequence
    ON {_RECOVERY_ACTION_TABLE} (
        {_RecoveryActionColumn.BATCH_ID},
        {_RecoveryActionColumn.SEQUENCE},
        {_RecoveryActionColumn.ACTION_RECORD_ID}
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_RECOVERY_ACTION_TABLE}_attempt_action
    ON {_RECOVERY_ACTION_TABLE} (
        {_RecoveryActionColumn.RECOVERY_ATTEMPT_ID},
        {_RecoveryActionColumn.ACTION_ID},
        {_RecoveryActionColumn.SEQUENCE},
        {_RecoveryActionColumn.ACTION_RECORD_ID}
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {_ARTIFACT_CLEANUP_TABLE} (
        {_ArtifactCleanupColumn.CLEANUP_RECORD_ID} TEXT PRIMARY KEY,
        {_ArtifactCleanupColumn.ARTIFACT_ID} TEXT NOT NULL,
        {_ArtifactCleanupColumn.BATCH_ID} TEXT NOT NULL REFERENCES {_BATCH_TABLE} ({_BatchColumn.BATCH_ID})
            ON DELETE CASCADE,
        {_ArtifactCleanupColumn.SEQUENCE} INTEGER NOT NULL,
        {_ArtifactCleanupColumn.TRIGGER} TEXT NOT NULL,
        {_ArtifactCleanupColumn.STATUS} TEXT NOT NULL,
        {_ArtifactCleanupColumn.RESOURCE_KEY} TEXT,
        {_ArtifactCleanupColumn.REASON} TEXT,
        {_ArtifactCleanupColumn.PAYLOAD} TEXT NOT NULL,
        {_ArtifactCleanupColumn.CREATED_AT} TEXT NOT NULL,
        UNIQUE ({_ArtifactCleanupColumn.BATCH_ID}, {_ArtifactCleanupColumn.SEQUENCE})
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_ARTIFACT_CLEANUP_TABLE}_batch_sequence
    ON {_ARTIFACT_CLEANUP_TABLE} (
        {_ArtifactCleanupColumn.BATCH_ID},
        {_ArtifactCleanupColumn.SEQUENCE},
        {_ArtifactCleanupColumn.CLEANUP_RECORD_ID}
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_ARTIFACT_CLEANUP_TABLE}_batch_artifact
    ON {_ARTIFACT_CLEANUP_TABLE} (
        {_ArtifactCleanupColumn.BATCH_ID},
        {_ArtifactCleanupColumn.ARTIFACT_ID},
        {_ArtifactCleanupColumn.SEQUENCE},
        {_ArtifactCleanupColumn.CLEANUP_RECORD_ID}
    )
    """,
)
