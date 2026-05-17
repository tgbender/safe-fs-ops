from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from safe_fs_ops.workspace import SafeWorkspaceError, _ClaimReleaseError
from safe_fs_ops.workspace_state import ClaimConflictError, LeaseLostError
from safe_fs_ops.workspace_transaction_state import remember_touched_operation_links

if TYPE_CHECKING:
    from safe_fs_ops.workspace_transaction import SafeTransaction


def resolve_operation_links(
    transaction: SafeTransaction,
    *,
    operation_run_id: str | None,
    operation_phase_id: str | None,
    now: datetime | None = None,
    create_implicit: bool = False,
) -> tuple[str | None, str | None]:
    with transaction._state_lock:
        transaction._require_entered()
        if operation_run_id is not None or operation_phase_id is not None:
            _validate_explicit_operation_links(
                transaction,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            remember_touched_operation_links(
                transaction,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            return operation_run_id, operation_phase_id
        if create_implicit:
            from safe_fs_ops.workspace_transaction_operation import ensure_implicit_operation_phase

            operation_run, operation_phase = ensure_implicit_operation_phase(transaction, now=now)
            remember_touched_operation_links(
                transaction,
                operation_run_id=operation_run.operation_run_id,
                operation_phase_id=operation_phase.operation_phase_id,
            )
            return operation_run.operation_run_id, operation_phase.operation_phase_id
        if transaction._operation_run is None or transaction._operation_phase is None:
            return None, None
        remember_touched_operation_links(
            transaction,
            operation_run_id=transaction._operation_run.operation_run_id,
            operation_phase_id=transaction._operation_phase.operation_phase_id,
        )
        return transaction._operation_run.operation_run_id, transaction._operation_phase.operation_phase_id


def _validate_explicit_operation_links(
    transaction: SafeTransaction,
    *,
    operation_run_id: str | None,
    operation_phase_id: str | None,
) -> None:
    if operation_run_id is None:
        if operation_phase_id is not None:
            raise ValueError("operation_phase_id requires operation_run_id")
        return
    operation_run = transaction.workspace.journal_store.get_operation_run(operation_run_id)
    if operation_run is None:
        raise ValueError(f"operation_run_id {operation_run_id!r} does not exist")
    if operation_run.run_id != transaction.run_id:
        raise ValueError(f"operation_run_id {operation_run_id!r} belongs to run_id {operation_run.run_id!r}")
    if operation_run.owner != transaction.workspace.owner:
        raise ValueError(f"operation_run_id {operation_run_id!r} belongs to owner {operation_run.owner!r}")
    _require_active_operation_link_status(
        "operation_run_id",
        operation_run_id,
        status=operation_run.status,
    )
    if operation_run.lease_name != transaction.lease.name:
        raise ValueError(f"operation_run_id {operation_run_id!r} belongs to lease {operation_run.lease_name!r}")
    if operation_run.lease_fencing_token != transaction.lease.fencing_token:
        raise ValueError(
            f"operation_run_id {operation_run_id!r} belongs to fencing token {operation_run.lease_fencing_token!r}"
        )
    if operation_phase_id is None:
        return
    operation_phase = transaction.workspace.journal_store.get_operation_phase(operation_phase_id)
    if operation_phase is None:
        raise ValueError(f"operation_phase_id {operation_phase_id!r} does not exist")
    if operation_phase.operation_run_id != operation_run_id:
        raise ValueError(
            f"operation_phase_id {operation_phase_id!r} belongs to operation_run_id "
            f"{operation_phase.operation_run_id!r}"
        )
    _require_active_operation_link_status(
        "operation_phase_id",
        operation_phase_id,
        status=operation_phase.status,
    )


def _require_active_operation_link_status(kind: str, record_id: str, *, status: str) -> None:
    if status == "active":
        return
    raise ValueError(f"{kind} {record_id!r} status {status!r} is terminal")


def cleanup_transaction(transaction: SafeTransaction) -> SafeWorkspaceError | None:
    from safe_fs_ops.workspace_transaction_operation import workspace_error

    lease = transaction._lease
    if lease is None:
        return None
    cleanup_now_value = cleanup_now(transaction)
    try:
        transaction._stop_lease_heartbeat()
    except LeaseLostError as exc:
        transaction._lease = None
        return workspace_error("transaction lease heartbeat failed", exc)
    try:
        transaction._release_claims_at(cleanup_now_value, tolerate_missing=True)
    except LeaseLostError as exc:
        transaction._lease = None
        return workspace_error("failed to release transaction claims", exc)
    except (ClaimConflictError, _ClaimReleaseError) as exc:
        transaction.workspace.lease_store.release(lease, now=cleanup_now_value)
        transaction._lease = None
        return workspace_error("failed to release transaction claims", exc)
    released = transaction.workspace.lease_store.release(lease, now=cleanup_now_value)
    transaction._lease = None
    if released:
        return None
    return SafeWorkspaceError("failed to release transaction lease")


def abandon_transaction_for_recovery(transaction: SafeTransaction) -> SafeWorkspaceError | None:
    """Stop local runtime state while leaving durable claims for recovery."""

    from safe_fs_ops.workspace_transaction_operation import workspace_error

    if transaction._lease is None:
        return None
    try:
        transaction._stop_lease_heartbeat()
    except LeaseLostError as exc:
        transaction._lease = None
        return workspace_error("transaction lease heartbeat failed", exc)
    transaction._lease = None
    return None


def cleanup_now(transaction: SafeTransaction) -> datetime:
    if transaction.cleanup_clock is None:
        from datetime import UTC

        return datetime.now(UTC)
    from safe_fs_ops.workspace_transaction_operation import normalized_utc

    return normalized_utc(transaction.cleanup_clock())


def record_cleanup_error(
    transaction: SafeTransaction, cleanup_error: SafeWorkspaceError | None, exc: BaseException
) -> None:
    del transaction
    if cleanup_error is None:
        return
    note = f"SafeWorkspace cleanup also failed: {cleanup_error}"
    exc.add_note(note)
