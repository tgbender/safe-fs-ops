from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG

from safe_fs_ops.filesystem_ops._windows_directory_operations import delete_file_windows
from safe_fs_ops.filesystem_ops.mutations import DeleteHooks, DurabilityMode, delete_file
from safe_fs_ops.operation_journal.filesystem_mutation_checkpoints import backup_content_path_for_state_path
from safe_fs_ops.operation_journal.models import CheckpointRecord


@dataclass(frozen=True, slots=True)
class BackupArtifactCleanupDebt:
    batch_id: str
    content_path: Path
    reason_code: str
    detail: str


@dataclass(frozen=True, slots=True)
class BackupArtifactCleanupCandidate:
    artifact_id: str
    batch_id: str
    checkpoint_id: str
    resource_key: str
    content_path: Path
    expected_content_path: Path | None
    expected_content_hash: str | None
    expected_size: int | None
    debt: BackupArtifactCleanupDebt | None = None


@dataclass(frozen=True, slots=True)
class BackupArtifactCleanupResult:
    artifact_id: str
    batch_id: str
    content_path: Path
    status: str
    reason_code: str | None = None
    detail: str | None = None


class BackupArtifactCleanupError(RuntimeError):
    def __init__(self, debts: Iterable[BackupArtifactCleanupDebt]) -> None:
        self.debts = tuple(debts)
        super().__init__(_cleanup_error_message(self.debts))


def cleanup_backup_artifacts(
    checkpoints: Iterable[CheckpointRecord],
    *,
    state_path: Path,
    delete_operation: Callable[[Path], None] | None = None,
) -> tuple[Path, ...]:
    delete_path = delete_operation
    deleted: list[Path] = []
    debts: list[BackupArtifactCleanupDebt] = []
    for candidate in plan_backup_artifact_cleanup_candidates(checkpoints, state_path=state_path):
        result = execute_backup_artifact_cleanup_candidate(
            candidate,
            state_path=state_path,
            delete_operation=delete_path,
        )
        if result.status == "succeeded":
            deleted.append(result.content_path)
            continue
        if result.status == "skipped":
            continue
        assert result.reason_code is not None
        assert result.detail is not None
        debts.append(
            BackupArtifactCleanupDebt(
                batch_id=result.batch_id,
                content_path=result.content_path,
                reason_code=result.reason_code,
                detail=result.detail,
            )
        )
    if debts:
        raise BackupArtifactCleanupError(debts)
    return tuple(deleted)


def backup_artifacts_root(state_path: Path) -> Path:
    return state_path.parent / f"{state_path.name}.artifacts"


@dataclass(frozen=True, slots=True)
class _ExpectedArtifactPath:
    batch_id: str
    checkpoint_id: str
    resource_key: str
    content_path: Path
    expected_content_path: Path | None
    expected_content_hash: str | None
    expected_size: int | None
    debt: BackupArtifactCleanupDebt | None = None


def plan_backup_artifact_cleanup_candidates(
    checkpoints: Iterable[CheckpointRecord],
    *,
    state_path: Path,
) -> tuple[BackupArtifactCleanupCandidate, ...]:
    paths = _backup_artifact_paths(checkpoints, state_path=state_path)
    seen_artifact_ids: set[str] = set()
    planned: list[BackupArtifactCleanupCandidate] = []
    for artifact in paths:
        artifact_id = _artifact_id_for_expected_path(artifact)
        if artifact_id in seen_artifact_ids:
            continue
        seen_artifact_ids.add(artifact_id)
        planned.append(
            BackupArtifactCleanupCandidate(
                artifact_id=artifact_id,
                batch_id=artifact.batch_id,
                checkpoint_id=artifact.checkpoint_id,
                resource_key=artifact.resource_key,
                content_path=artifact.content_path,
                expected_content_path=artifact.expected_content_path,
                expected_content_hash=artifact.expected_content_hash,
                expected_size=artifact.expected_size,
                debt=artifact.debt,
            )
        )
    return tuple(planned)


def execute_backup_artifact_cleanup_candidate(
    candidate: BackupArtifactCleanupCandidate,
    *,
    state_path: Path,
    delete_operation: Callable[[Path], None] | None = None,
    before_delete: Callable[[], None] | None = None,
) -> BackupArtifactCleanupResult:
    artifacts_root = backup_artifacts_root(state_path)
    delete_path = delete_operation
    if candidate.debt is not None:
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code=candidate.debt.reason_code,
            detail=candidate.debt.detail,
        )
    if candidate.expected_content_path is None:
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="malformed_backup_checkpoint",
            detail="backup artifact cleanup candidate is missing expected content path",
        )
    if candidate.expected_content_hash is None or candidate.expected_size is None:
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="malformed_backup_checkpoint",
            detail="backup artifact cleanup candidate is missing expected content hash or size",
        )
    if not _matches_artifact_path_contract(candidate.content_path, candidate.expected_content_path):
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="unsafe_artifact_path",
            detail=(
                "refusing to remove backup artifact with unexpected path "
                f"{candidate.content_path}; expected {candidate.expected_content_path}"
            ),
        )
    symlink_detail = _symlink_chain_violation(candidate.content_path, artifacts_root)
    if symlink_detail is not None:
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="unsafe_artifact_path",
            detail=symlink_detail,
        )
    verification = _verify_backup_artifact_cleanup_candidate(candidate)
    if verification is not None:
        return verification
    if before_delete is not None:
        before_delete()
        verification = _verify_backup_artifact_cleanup_candidate(candidate)
        if verification is not None:
            return verification
    try:
        if delete_path is None:
            _default_delete_verified_operation(
                candidate.content_path,
                verify=lambda: _verify_backup_artifact_cleanup_candidate(candidate),
            )
        else:
            delete_path(candidate.content_path)
    except _BackupArtifactVerificationChanged as exc:
        return exc.result
    except FileNotFoundError:
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="skipped",
            reason_code="artifact_missing",
            detail=f"backup artifact already missing at {candidate.content_path}",
        )
    except OSError as exc:
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="failed",
            reason_code="artifact_cleanup_failed",
            detail=str(exc),
        )
    return BackupArtifactCleanupResult(
        artifact_id=candidate.artifact_id,
        batch_id=candidate.batch_id,
        content_path=candidate.content_path,
        status="succeeded",
    )


def _verify_backup_artifact_cleanup_candidate(
    candidate: BackupArtifactCleanupCandidate,
) -> BackupArtifactCleanupResult | None:
    assert candidate.expected_content_hash is not None
    assert candidate.expected_size is not None
    try:
        actual_hash, actual_size = _hash_regular_file(candidate.content_path)
    except FileNotFoundError:
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="skipped",
            reason_code="artifact_missing",
            detail=f"backup artifact already missing at {candidate.content_path}",
        )
    except OSError as exc:
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="backup_artifact_verification_failed",
            detail=str(exc),
        )
    if actual_hash != candidate.expected_content_hash or actual_size != candidate.expected_size:
        return BackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="backup_artifact_content_mismatch",
            detail=f"backup artifact content changed at {candidate.content_path}",
        )
    return None


def _backup_artifact_paths(
    checkpoints: Iterable[CheckpointRecord],
    *,
    state_path: Path,
) -> tuple[_ExpectedArtifactPath, ...]:
    checkpoint_records = tuple(checkpoints)
    backup_checkpoint_keys = {
        _backup_operation_key(checkpoint) for checkpoint in checkpoint_records if checkpoint.checkpoint_type == "backup"
    }
    paths: list[_ExpectedArtifactPath] = []
    for checkpoint in checkpoint_records:
        if checkpoint.checkpoint_type not in {"backup", "backup_artifact_intent"}:
            continue
        if (
            checkpoint.checkpoint_type == "backup_artifact_intent"
            and _backup_operation_key(checkpoint) in backup_checkpoint_keys
        ):
            continue
        if _is_missing_backup_without_artifact(checkpoint):
            continue
        content_path = checkpoint.payload.get("content_path")
        source_path = checkpoint.payload.get("path")
        operation_id = checkpoint.operation_id
        missing_fields: list[str] = []
        if content_path is None:
            missing_fields.append("content_path")
        if source_path is None:
            missing_fields.append("path")
        if operation_id is None:
            missing_fields.append("operation_id")
        expected_content_hash = _optional_str(checkpoint.payload.get("content_hash"))
        expected_size = _optional_int(checkpoint.payload.get("size"))
        if not _is_missing_backup_without_artifact(checkpoint):
            if expected_content_hash is None:
                missing_fields.append("content_hash")
            if expected_size is None:
                missing_fields.append("size")
        if missing_fields:
            expected_content_path = (
                backup_content_path_for_state_path(
                    state_path,
                    batch_id=checkpoint.batch_id,
                    operation_id=operation_id,
                    path=Path(str(source_path)),
                )
                if source_path is not None and operation_id is not None
                else None
            )
            debt_path = (
                Path(str(content_path))
                if content_path is not None
                else expected_content_path
                if expected_content_path is not None
                else Path(str(source_path))
                if source_path is not None
                else backup_artifacts_root(state_path)
            )
            paths.append(
                _ExpectedArtifactPath(
                    batch_id=checkpoint.batch_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    resource_key=checkpoint.resource_key,
                    content_path=debt_path,
                    expected_content_path=expected_content_path,
                    expected_content_hash=expected_content_hash,
                    expected_size=expected_size,
                    debt=BackupArtifactCleanupDebt(
                        batch_id=checkpoint.batch_id,
                        content_path=debt_path,
                        reason_code="malformed_backup_checkpoint",
                        detail=("backup checkpoint is missing required field(s): " + ", ".join(missing_fields)),
                    ),
                )
            )
            continue
        assert content_path is not None
        assert source_path is not None
        assert operation_id is not None
        expected_content_hash = _optional_str(checkpoint.payload.get("content_hash"))
        expected_size = _optional_int(checkpoint.payload.get("size"))
        content = Path(str(content_path))
        expected_content_path = _expected_backup_content_path_for_checkpoint(
            state_path,
            batch_id=checkpoint.batch_id,
            operation_id=operation_id,
            path=Path(str(source_path)),
            content_path=content,
        )
        paths.append(
            _ExpectedArtifactPath(
                batch_id=checkpoint.batch_id,
                checkpoint_id=checkpoint.checkpoint_id,
                resource_key=checkpoint.resource_key,
                content_path=content,
                expected_content_path=expected_content_path,
                expected_content_hash=expected_content_hash,
                expected_size=expected_size,
            )
        )
    return tuple(paths)


def _expected_backup_content_path_for_checkpoint(
    state_path: Path,
    *,
    batch_id: str,
    operation_id: str,
    path: Path,
    content_path: Path,
) -> Path:
    expected_content_path = backup_content_path_for_state_path(
        state_path,
        batch_id=batch_id,
        operation_id=operation_id,
        path=path,
    )
    legacy_content_path = _legacy_backup_content_path_for_state_path(
        state_path,
        batch_id=batch_id,
        operation_id=operation_id,
        path=path,
    )
    if _matches_artifact_path_contract(content_path, legacy_content_path):
        return legacy_content_path
    return expected_content_path


def _legacy_backup_content_path_for_state_path(
    state_path: Path,
    *,
    batch_id: str,
    operation_id: str,
    path: Path,
) -> Path:
    extension = "".join(path.suffixes)
    artifact_name = operation_id if not extension else f"{operation_id}{extension}.bak"
    if extension == "":
        artifact_name = f"{operation_id}.bak"
    return (
        state_path.parent
        / f"{state_path.name}.artifacts"
        / "file-backups"
        / batch_id
        / uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex
        / artifact_name
    )


class _BackupArtifactVerificationChanged(RuntimeError):
    def __init__(self, result: BackupArtifactCleanupResult) -> None:
        super().__init__(result.detail)
        self.result = result


def _backup_operation_key(checkpoint: CheckpointRecord) -> tuple[str, str | None, str | None]:
    path = checkpoint.payload.get("path")
    return (
        checkpoint.batch_id,
        checkpoint.operation_id,
        None if path is None else str(path),
    )


def _is_missing_backup_without_artifact(checkpoint: CheckpointRecord) -> bool:
    payload = checkpoint.payload
    if payload.get("content_path") is not None:
        return False
    if payload.get("existed") is not False:
        return False

    file_type = payload.get("file_type")
    if file_type not in (None, "missing"):
        return False

    snapshot = payload.get("snapshot")
    if snapshot is None:
        return True
    if not isinstance(snapshot, Mapping):
        return False

    snapshot_exists = snapshot.get("exists")
    if snapshot_exists not in (None, False):
        return False

    snapshot_file_type = snapshot.get("file_type")
    return snapshot_file_type in (None, "missing")


def _default_delete_operation(path: Path) -> None:
    if os.name == "nt":
        delete_file_windows(path, durability=DurabilityMode.FSYNC)
        return
    delete_file(path, durability=DurabilityMode.FSYNC)


def _default_delete_verified_operation(
    path: Path,
    *,
    verify: Callable[[], BackupArtifactCleanupResult | None],
) -> None:
    def after_unlink_validation(_path: Path) -> None:
        result = verify()
        if result is not None:
            raise _BackupArtifactVerificationChanged(result)

    hooks = DeleteHooks(after_unlink_validation=after_unlink_validation)
    if os.name == "nt":
        delete_file_windows(path, durability=DurabilityMode.FSYNC, hooks=hooks)
        return
    delete_file(path, durability=DurabilityMode.FSYNC, hooks=hooks)


def _hash_regular_file(path: Path) -> tuple[str, int]:
    stat_result = path.lstat()
    if not S_ISREG(stat_result.st_mode):
        raise OSError(f"backup artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _matches_artifact_path_contract(content_path: Path, expected_content_path: Path) -> bool:
    try:
        return _normalized_path(content_path) == _normalized_path(expected_content_path)
    except OSError:
        return False


def _symlink_chain_violation(path: Path, root: Path) -> str | None:
    normalized_path = _normalized_path(path)
    normalized_root = _normalized_path(root)
    try:
        normalized_path.relative_to(normalized_root)
    except ValueError:
        return f"refusing to remove backup artifact outside {root}"
    for candidate in _path_chain(normalized_root, normalized_path):
        try:
            if candidate.is_symlink():
                return f"refusing to remove backup artifact beneath symlinked path {candidate}"
        except OSError as exc:
            return f"refusing to inspect backup artifact path {candidate}: {exc}"
    return None


def _path_chain(root: Path, path: Path) -> tuple[Path, ...]:
    chain: list[Path] = [path]
    chain.extend(path.parents)
    included: list[Path] = []
    for candidate in chain:
        included.append(candidate)
        if candidate == root:
            break
    return tuple(reversed(included))


def _normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _artifact_id_for_expected_path(artifact: _ExpectedArtifactPath) -> str:
    if artifact.expected_content_path is not None:
        return str(_normalized_path(artifact.expected_content_path))
    return f"checkpoint:{artifact.checkpoint_id}"


def _cleanup_error_message(debts: tuple[BackupArtifactCleanupDebt, ...]) -> str:
    if len(debts) == 1:
        debt = debts[0]
        return (
            f"backup artifact cleanup debt for batch {debt.batch_id!r}: "
            f"{debt.reason_code} at {debt.content_path} ({debt.detail})"
        )
    details = ", ".join(f"{debt.batch_id!r}:{debt.reason_code}:{debt.content_path}" for debt in debts)
    return f"backup artifact cleanup debt for {len(debts)} batches: {details}"


__all__ = [
    "BackupArtifactCleanupDebt",
    "BackupArtifactCleanupCandidate",
    "BackupArtifactCleanupError",
    "BackupArtifactCleanupResult",
    "backup_artifacts_root",
    "cleanup_backup_artifacts",
    "execute_backup_artifact_cleanup_candidate",
    "plan_backup_artifact_cleanup_candidates",
]
