from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from safe_fs_ops import SafeWorkspace
from safe_fs_ops.filesystem_ops import snapshot_resource
from safe_fs_ops.filesystem_ops._windows_operations import (
    atomic_write_text_windows,
    delete_file_windows,
    make_directory_windows,
)
from safe_fs_ops.workspace import _detect_workspace_backend

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.safe_fs_ops_backend]


def test_workspace_reports_default_backend_capabilities(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")

    backend = workspace.filesystem_backend

    assert backend.is_default_backend is True
    if os.name == "nt":
        assert backend.backend_name == "windows_native"
    else:
        assert backend.backend_name == "posix_descriptor_relative"
    assert backend.platform in {"windows", "posix", "macos"}
    assert backend.support_for("write_text") == backend.write_text
    assert backend.support_for("delete_file") == backend.delete_file
    assert backend.support_for("make_directory") == backend.make_directory
    assert backend.support_for("remove_directory") == backend.remove_directory


@pytest.mark.skipif(os.name != "nt", reason="native Windows default backend characterization")
def test_windows_default_backend_reports_windows_native_backend_name(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")

    backend = workspace.filesystem_backend

    assert backend.is_default_backend is True
    assert backend.platform == "windows"
    assert backend.backend_name == "windows_native"


@pytest.mark.skipif(os.name == "nt", reason="POSIX default backend characterization")
def test_posix_default_backend_reports_posix_descriptor_relative_backend_name(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")

    backend = workspace.filesystem_backend

    assert backend.is_default_backend is True
    assert backend.backend_name == "posix_descriptor_relative"
    assert backend.platform == ("macos" if sys.platform == "darwin" else "posix")


def test_workspace_reports_custom_backend_when_filesystem_operation_is_injected(tmp_path: Path) -> None:
    def custom_make_directory(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        Path(path).mkdir(parents=parents, exist_ok=exist_ok)

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        make_directory_operation=custom_make_directory,
    )

    backend = workspace.filesystem_backend

    assert backend.is_default_backend is False
    assert backend.backend_name == "custom"
    assert backend.write_text.state == "unknown"
    assert backend.delete_file.state == "unknown"
    assert backend.make_directory.state == "unknown"
    assert backend.remove_directory.state == "unknown"


@pytest.mark.platform_windows
def test_workspace_reports_custom_backend_for_mixed_default_posix_and_windows_tuple(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    workspace.snapshot = snapshot_resource

    backend = _detect_workspace_backend(workspace)

    assert backend.is_default_backend is False
    assert backend.backend_name == "custom"


@pytest.mark.skipif(os.name != "nt", reason="native Windows default backend characterization")
def test_windows_default_backend_uses_windows_native_mutations(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "config.txt"
    nested = tmp_path / "workspace" / "nested" / "state"

    assert workspace.write_text_operation is atomic_write_text_windows
    assert workspace.delete_file_operation is delete_file_windows
    assert workspace.make_directory_operation is make_directory_windows
    assert workspace.filesystem_backend.write_text.supported is True
    assert workspace.filesystem_backend.delete_file.supported is True
    assert workspace.filesystem_backend.make_directory.supported is True

    with workspace.transaction(name="write", resources={"file": workspace.file(target)}) as transaction:
        transaction.write_text(transaction.r.file, "enabled = true\n")

    assert target.read_text(encoding="utf-8") == "enabled = true\n"

    with workspace.transaction(name="mkdir", resources={"dir": workspace.directory(nested)}) as transaction:
        mkdir_result = transaction.make_directory(transaction.r.dir, parents=True, exist_ok=True)

    assert mkdir_result.skipped is False
    assert nested.is_dir()

    with workspace.transaction(name="delete", resources={"file": workspace.file(target)}) as transaction:
        transaction.delete_file(transaction.r.file)

    assert target.exists() is False


@pytest.mark.skipif(os.name != "nt", reason="native Windows rollback characterization")
@pytest.mark.slow_recovery
def test_windows_default_backend_automatic_rollback_removes_created_directory(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "nested" / "state"
    resources = workspace.resources({"dir": workspace.directory(target)})

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="mkdir",
            resources=resources,
            run_id="run-1",
            rollback="automatic",
        ) as transaction,
    ):
        transaction.make_directory(transaction.r.dir, parents=True, exist_ok=True)
        raise RuntimeError("boom")

    assert target.exists() is False


@pytest.mark.skipif(os.name != "nt", reason="native Windows file rollback characterization")
@pytest.mark.slow_recovery
def test_windows_default_backend_automatic_rollback_removes_created_file(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "config.txt"

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="write-create",
            resources={"file": workspace.file(target)},
            rollback="automatic",
        ) as transaction,
    ):
        transaction.write_text(transaction.r.file, "created\n")
        raise RuntimeError("boom")

    assert target.exists() is False


@pytest.mark.skipif(os.name != "nt", reason="native Windows file rollback characterization")
@pytest.mark.slow_recovery
def test_windows_default_backend_automatic_rollback_restores_overwritten_file(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "config.txt"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="write-overwrite",
            resources={"file": workspace.file(target)},
            rollback="automatic",
        ) as transaction,
    ):
        transaction.write_text(transaction.r.file, "replacement\n")
        raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "original\n"


@pytest.mark.skipif(os.name != "nt", reason="native Windows file rollback characterization")
@pytest.mark.slow_recovery
def test_windows_default_backend_automatic_rollback_restores_deleted_file(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "config.txt"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="delete",
            resources={"file": workspace.file(target)},
            rollback="automatic",
        ) as transaction,
    ):
        transaction.delete_file(transaction.r.file)
        raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "original\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX default backend file rollback characterization")
@pytest.mark.slow_recovery
def test_posix_default_backend_automatic_rollback_removes_created_file(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "config.txt"
    target.parent.mkdir()

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="write-create",
            resources={"file": workspace.file(target)},
            rollback="automatic",
        ) as transaction,
    ):
        transaction.write_text(transaction.r.file, "created\n")
        raise RuntimeError("boom")

    assert target.exists() is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX default backend file rollback characterization")
@pytest.mark.slow_recovery
def test_posix_default_backend_automatic_rollback_restores_overwritten_file(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "config.txt"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="write-overwrite",
            resources={"file": workspace.file(target)},
            rollback="automatic",
        ) as transaction,
    ):
        transaction.write_text(transaction.r.file, "replacement\n")
        raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "original\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX default backend file rollback characterization")
@pytest.mark.slow_recovery
def test_posix_default_backend_automatic_rollback_restores_deleted_file(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "config.txt"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="delete",
            resources={"file": workspace.file(target)},
            rollback="automatic",
        ) as transaction,
    ):
        transaction.delete_file(transaction.r.file)
        raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "original\n"


@pytest.mark.skipif(os.name == "nt", reason="default backend support is characterized separately on native Windows")
def test_posix_default_backend_write_text_and_delete_file_succeed(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "config.txt"
    target.parent.mkdir()

    assert workspace.filesystem_backend.write_text.supported is True
    assert workspace.filesystem_backend.delete_file.supported is True

    with workspace.transaction(name="write", resources={"file": workspace.file(target)}) as transaction:
        transaction.write_text(transaction.r.file, "enabled = true\n")

    assert target.read_text(encoding="utf-8") == "enabled = true\n"

    with workspace.transaction(name="delete", resources={"file": workspace.file(target)}) as transaction:
        transaction.delete_file(transaction.r.file)

    assert target.exists() is False


@pytest.mark.skipif(os.name == "nt", reason="default backend support is characterized separately on native Windows")
def test_posix_default_backend_recursive_mkdir_and_existing_dir_noop(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "workspace" / "nested" / "state"

    assert workspace.filesystem_backend.make_directory.supported is True

    with workspace.transaction(name="mkdir", resources={"dir": workspace.directory(target)}) as transaction:
        result = transaction.make_directory(transaction.r.dir, parents=True, exist_ok=True)

    assert result.skipped is False
    assert target.is_dir()

    with workspace.transaction(name="mkdir-noop", resources={"dir": workspace.directory(target)}) as transaction:
        noop_result = transaction.make_directory(transaction.r.dir, parents=True, exist_ok=True)

    assert noop_result.skipped is True
    assert target.is_dir()
