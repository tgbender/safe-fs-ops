from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from safe_fs_ops.workspace_state.models import ClaimRecord

if TYPE_CHECKING:
    from safe_fs_ops.workspace_transaction import SafeTransaction


def remember_claim(transaction: SafeTransaction, claim: ClaimRecord) -> None:
    with transaction._state_lock:
        transaction._claims_by_resource_key[claim.resource_key] = claim
        if claim.resource_key not in transaction._claim_release_order:
            transaction._claim_release_order.append(claim.resource_key)


def remember_touched_operation_links(
    transaction: SafeTransaction,
    *,
    operation_run_id: str | None,
    operation_phase_id: str | None,
) -> None:
    with transaction._state_lock:
        if operation_run_id is not None:
            transaction._touched_operation_run_ids.add(operation_run_id)
        if operation_phase_id is not None:
            transaction._touched_operation_phase_ids.add(operation_phase_id)


def forget_claim(transaction: SafeTransaction, resource_key: str) -> None:
    with transaction._state_lock:
        transaction._claims_by_resource_key.pop(resource_key, None)
        with suppress(ValueError):
            transaction._claim_release_order.remove(resource_key)


def claim_is_already_released(transaction: SafeTransaction, resource_key: str) -> bool:
    claim_getter = getattr(transaction.workspace.claim_store, "get", None)
    if claim_getter is None:
        return False
    return claim_getter(resource_key) is None


def close_from_operation_exit(transaction: SafeTransaction) -> None:
    with transaction._state_lock:
        transaction._entered = False
        transaction._closed = True
