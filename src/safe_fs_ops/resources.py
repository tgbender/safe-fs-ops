from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from safe_fs_ops.filesystem_ops import PathSafety, ResourceSnapshot, inspect_path, snapshot_resource
from safe_fs_ops.filesystem_ops.paths import absolute_without_resolving
from safe_fs_ops.operation_journal import directory_resource_key, file_resource_key, tree_resource_key

ResourceKind = Literal["file", "directory", "tree", "custom"]


@dataclass(frozen=True, slots=True)
class ResourceHandle:
    original_path: Path | None
    path: Path | None
    resource_key: str
    kind: ResourceKind
    label: str | None = None
    scope: str | None = None

    def with_label(self, label: str) -> ResourceHandle:
        return replace(self, label=label)

    def inspect(self) -> PathSafety:
        if self.path is None:
            raise ValueError("custom resources without paths cannot be inspected")
        return inspect_path(self.path)

    def snapshot(self) -> ResourceSnapshot:
        if self.path is None:
            raise ValueError("custom resources without paths cannot be snapshotted")
        return snapshot_resource(self.path)


@dataclass(frozen=True, slots=True)
class FileResource(ResourceHandle):
    kind: Literal["file"] = "file"


@dataclass(frozen=True, slots=True)
class DirectoryResource(ResourceHandle):
    kind: Literal["directory"] = "directory"

    def file(self, relative_path: Path | str, *, label: str | None = None) -> FileResource:
        return file_resource(_child_path(self, relative_path), label=label, scope=self.scope)

    def directory(self, relative_path: Path | str, *, label: str | None = None) -> DirectoryResource:
        return directory_resource(_child_path(self, relative_path), label=label, scope=self.scope)

    def tree(self, relative_path: Path | str, *, label: str | None = None) -> TreeResource:
        return tree_resource(_child_path(self, relative_path), label=label, scope=self.scope)


@dataclass(frozen=True, slots=True)
class TreeResource(ResourceHandle):
    kind: Literal["tree"] = "tree"

    def file(self, relative_path: Path | str, *, label: str | None = None) -> FileResource:
        return file_resource(_child_path(self, relative_path), label=label, scope=self.scope)

    def directory(self, relative_path: Path | str, *, label: str | None = None) -> DirectoryResource:
        return directory_resource(_child_path(self, relative_path), label=label, scope=self.scope)

    def tree(self, relative_path: Path | str, *, label: str | None = None) -> TreeResource:
        return tree_resource(_child_path(self, relative_path), label=label, scope=self.scope)


class ResourceSet(Mapping[str, ResourceHandle]):
    def __init__(self, resources: Mapping[str, ResourceHandle]) -> None:
        labeled: dict[str, ResourceHandle] = {}
        seen_keys: dict[str, str] = {}
        for name, resource in resources.items():
            if not name:
                raise ValueError("resource names must be non-empty")
            if not isinstance(resource, ResourceHandle):
                raise TypeError(f"resource {name!r} must be a ResourceHandle")
            existing_name = seen_keys.get(resource.resource_key)
            if existing_name is not None:
                raise ValueError(f"resources {existing_name!r} and {name!r} both refer to {resource.resource_key!r}")
            seen_keys[resource.resource_key] = name
            labeled[name] = resource.with_label(name) if resource.label is None else resource
        self._resources = MappingProxyType(dict(labeled))
        self._keys_by_resource_key = MappingProxyType(
            {resource.resource_key: name for name, resource in labeled.items()}
        )

    def __getitem__(self, key: str) -> ResourceHandle:
        return self._resources[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._resources)

    def __len__(self) -> int:
        return len(self._resources)

    def __getattr__(self, name: str) -> ResourceHandle:
        try:
            return self._resources[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def sorted(self) -> tuple[ResourceHandle, ...]:
        return tuple(sorted(self._resources.values(), key=lambda resource: resource.resource_key))

    def contains_resource(self, resource: ResourceHandle) -> bool:
        return any(stored is resource for stored in self._resources.values())

    def contains_resource_key(self, resource_key: str) -> bool:
        return self._keys_by_resource_key.get(resource_key) is not None


def file_resource(path: Path | str, *, label: str | None = None, scope: str | None = None) -> FileResource:
    original = Path(path).expanduser()
    canonical = absolute_without_resolving(original)
    return FileResource(
        original_path=original,
        path=canonical,
        resource_key=file_resource_key(canonical),
        label=label,
        scope=scope,
    )


def directory_resource(
    path: Path | str,
    *,
    label: str | None = None,
    scope: str | None = None,
) -> DirectoryResource:
    original = Path(path).expanduser()
    canonical = absolute_without_resolving(original)
    return DirectoryResource(
        original_path=original,
        path=canonical,
        resource_key=directory_resource_key(canonical),
        label=label,
        scope=scope,
    )


def tree_resource(path: Path | str, *, label: str | None = None, scope: str | None = None) -> TreeResource:
    original = Path(path).expanduser()
    canonical = absolute_without_resolving(original)
    return TreeResource(
        original_path=original,
        path=canonical,
        resource_key=tree_resource_key(canonical),
        label=label,
        scope=scope,
    )


def custom_resource(resource_key: str, *, label: str | None = None, scope: str | None = None) -> ResourceHandle:
    if not resource_key:
        raise ValueError("resource_key must be non-empty")
    return ResourceHandle(
        original_path=None,
        path=None,
        resource_key=resource_key,
        kind="custom",
        label=label,
        scope=scope,
    )


def _child_path(parent: ResourceHandle, relative_path: Path | str) -> Path:
    if parent.path is None:
        raise ValueError("custom resources cannot create child path resources")
    child = Path(relative_path)
    if child.is_absolute():
        raise ValueError("child resource paths must be relative")
    candidate = absolute_without_resolving(parent.path / child)
    try:
        candidate.relative_to(parent.path)
    except ValueError as exc:
        raise ValueError("child resource paths must stay within the parent resource") from exc
    return candidate


__all__ = [
    "DirectoryResource",
    "FileResource",
    "ResourceHandle",
    "ResourceKind",
    "ResourceSet",
    "TreeResource",
    "custom_resource",
    "directory_resource",
    "file_resource",
    "tree_resource",
]
