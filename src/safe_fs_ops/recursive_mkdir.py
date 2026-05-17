from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from safe_fs_ops.filesystem_ops.models import PathSafety
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, absolute_without_resolving, inspect_path
from safe_fs_ops.operation_journal.filesystem_support import directory_resource_key


@dataclass(frozen=True, slots=True)
class RecursiveMkdirStep:
    path: Path
    resource_key: str
    existed_before: bool
    cleanup_eligible: bool


@dataclass(frozen=True, slots=True)
class RecursiveMkdirPlan:
    target: Path
    parents: bool
    exist_ok: bool
    root_boundary: Path | None
    target_existed_before: bool
    steps: tuple[RecursiveMkdirStep, ...]


def plan_directory_creation(
    path: Path | str,
    *,
    parents: bool = False,
    exist_ok: bool = False,
    root_boundary: Path | str | None = None,
    path_inspector: Callable[[Path], PathSafety] = inspect_path,
) -> RecursiveMkdirPlan:
    target = absolute_without_resolving(path)
    boundary = _canonical_boundary(root_boundary)
    if boundary is not None:
        _require_path_within_boundary(target, boundary)
        _require_existing_directory(
            boundary,
            role="root boundary",
            operation="mkdir plan",
            path_inspector=path_inspector,
        )

    target_safety = path_inspector(target)
    if target_safety.exists:
        _require_safe_existing_directory(
            target,
            role="target",
            operation="mkdir plan",
            path_inspector=path_inspector,
        )
        if exist_ok:
            return RecursiveMkdirPlan(
                target=target,
                parents=parents,
                exist_ok=exist_ok,
                root_boundary=boundary,
                target_existed_before=True,
                steps=(),
            )
        raise FileExistsError(target)

    missing_chain = _collect_missing_chain(target, boundary=boundary, path_inspector=path_inspector)
    if not parents and len(missing_chain) != 1:
        raise FileNotFoundError(target.parent)

    ordered_steps = tuple(
        RecursiveMkdirStep(
            path=missing_path,
            resource_key=directory_resource_key(missing_path),
            existed_before=False,
            cleanup_eligible=False,
        )
        for missing_path in reversed(missing_chain)
    )
    return RecursiveMkdirPlan(
        target=target,
        parents=parents,
        exist_ok=exist_ok,
        root_boundary=boundary,
        target_existed_before=False,
        steps=ordered_steps,
    )


def _collect_missing_chain(
    target: Path,
    *,
    boundary: Path | None,
    path_inspector: Callable[[Path], PathSafety],
) -> list[Path]:
    missing_chain: list[Path] = []
    current = target
    while True:
        safety = path_inspector(current)
        if safety.exists:
            _require_safe_existing_directory(
                current,
                role="ancestor",
                operation="mkdir plan",
                path_inspector=path_inspector,
            )
            return missing_chain
        missing_chain.append(current)
        if boundary is not None and current == boundary:
            raise FileNotFoundError(boundary)
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(current)
        current = parent


def _canonical_boundary(root_boundary: Path | str | None) -> Path | None:
    if root_boundary is None:
        return None
    return absolute_without_resolving(root_boundary)


def _require_path_within_boundary(path: Path, boundary: Path) -> None:
    try:
        path.relative_to(boundary)
    except ValueError as exc:
        raise UnsafePathError(f"mkdir plan refused because target escapes root boundary: {path}") from exc


def _require_existing_directory(
    path: Path,
    *,
    role: str,
    operation: str,
    path_inspector: Callable[[Path], PathSafety],
) -> None:
    safety = path_inspector(path)
    if not safety.exists:
        raise FileNotFoundError(path)
    _require_safe_existing_directory(path, role=role, operation=operation, path_inspector=path_inspector)


def _require_safe_existing_directory(
    path: Path,
    *,
    role: str,
    operation: str,
    path_inspector: Callable[[Path], PathSafety],
) -> None:
    safety = path_inspector(path)
    if safety.is_symlink:
        raise UnsafePathError(f"{operation} refused because {role} is a symlink: {path}")
    if safety.is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused because {role} is a Windows reparse point: {path}")
    if safety.is_mount:
        raise UnsafePathError(f"{operation} refused because {role} is a mount point: {path}")
    if not safety.is_dir:
        raise UnsafePathError(f"{operation} refused because {role} is not a directory: {path}")
