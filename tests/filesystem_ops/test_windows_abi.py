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


def test_ctypes_layouts_match_compiled_windows_sdk(tmp_path: Path) -> None:
    shutil.copyfile(Path(__file__).with_name("windows_abi_probe.c"), tmp_path / "probe.c")
    command = ["cl", "/nologo", "/W4", "probe.c", "/Fe:probe.exe"]
    if shutil.which("cl") is None:
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
        command = [
            "cmd.exe",
            "/d",
            "/s",
            "/c",
            f'call "{developer_shell}" -no_logo -arch={arch} -host_arch=amd64 && cl /nologo /W4 probe.c /Fe:probe.exe',
        ]
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
