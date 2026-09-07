import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import _windows_identity_rmdir as remove
from safe_fs_ops.filesystem_ops import no_replace_rename as rename

pytestmark = [pytest.mark.platform_windows, pytest.mark.skipif(os.name != "nt", reason="native Windows SDK")]


@pytest.mark.parametrize("initialize_msvc", [False, True])
def test_ctypes_layouts_match_compiled_windows_sdk(tmp_path: Path, initialize_msvc: bool) -> None:
    shutil.copyfile(Path(__file__).with_name("windows_abi_probe.c"), tmp_path / "probe.c")
    command = ["cl", "/nologo", "/W4", "probe.c", "/Fe:probe.exe"]
    if initialize_msvc or shutil.which("cl") is None:
        vswhere = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / (
            "Microsoft Visual Studio/Installer/vswhere.exe"
        )
        installation = ""
        if vswhere.exists():
            installation = subprocess.check_output(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                text=True,
                timeout=30,
            ).strip()
        if not installation:
            if os.environ.get("CI"):
                pytest.fail("Windows SDK compiler is required for CI ABI verification")
            pytest.skip("Windows SDK compiler is not installed")
        developer_shell = Path(installation) / "Common7/Tools/VsDevCmd.bat"
        arch = "amd64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "x86"
        command = _msvc_compile_command(tmp_path, developer_shell, arch)
    compilation = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
    sdk = json.loads(subprocess.check_output([str(tmp_path / "probe.exe")], text=True, timeout=10))
    observed = dict(
        handle=ctypes.sizeof(ctypes.c_void_p),
        wchar=ctypes.sizeof(ctypes.c_wchar),
        by_handle_info=ctypes.sizeof(remove._ByHandleFileInformation),
        disposition=ctypes.sizeof(remove._FileDispositionInfo),
        disposition_ex=ctypes.sizeof(remove._FileDispositionInfoEx),
        io_status=ctypes.sizeof(rename._IoStatusBlock),
        io_information_offset=rename._IoStatusBlock.Information.offset,
        rename_root_offset=rename._FileRenameInfoTemplate.RootDirectory.offset,
        rename_length_offset=rename._FileRenameInfoTemplate.FileNameLength.offset,
        rename_name_offset=rename._FileRenameInfoTemplate.FileName.offset,
    )
    assert observed == sdk


@pytest.mark.parametrize(("setup_exit", "compiler_exit"), [(0, 0), (0, 7), (9, 0)])
def test_msvc_launcher_handles_spaces_and_propagates_errors(
    tmp_path: Path, setup_exit: int, compiler_exit: int
) -> None:
    tools = tmp_path / "Compiler Tools"
    tools.mkdir()
    developer_shell = tools / "Developer Shell.bat"
    developer_shell.write_text(f'@echo off\nset "PATH=%~dp0;%PATH%"\nexit /b {setup_exit}\n', encoding="utf-8")
    (tools / "cl.cmd").write_text(
        f"@echo off\n> compiler-args.txt echo %*\nexit /b {compiler_exit}\n", encoding="utf-8"
    )
    result = subprocess.run(
        _msvc_compile_command(tmp_path, developer_shell, "amd64"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (setup_exit or compiler_exit), result.stdout + result.stderr
    arguments = tmp_path / "compiler-args.txt"
    if setup_exit:
        assert not arguments.exists()
    else:
        assert arguments.read_text().strip() == "/nologo /W4 probe.c /Fe:probe.exe"


def _msvc_compile_command(workdir: Path, developer_shell: Path, arch: str) -> list[str]:
    # A list argument containing a quoted command is escaped for the C runtime,
    # not cmd.exe. Keep shell syntax in a batch file and pass its simple name.
    script = workdir / "compile-probe.cmd"
    script.write_text(
        "@echo off\n"
        f'call "{developer_shell}" -no_logo -arch={arch} -host_arch=amd64\n'
        "if errorlevel 1 exit /b %errorlevel%\n"
        "cl /nologo /W4 probe.c /Fe:probe.exe\n",
        encoding="utf-8",
    )
    return ["cmd.exe", "/d", "/c", script.name]
