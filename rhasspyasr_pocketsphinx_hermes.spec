# -*- mode: python -*-
import os
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

# Use either virtual environment or lib/bin dirs from environment variables
venv = Path.cwd() / ".venv"
venv_lib = venv / "lib"
for dir_path in venv_lib.glob("python*"):
    if dir_path.is_dir() and (dir_path / "site-packages").exists():
        site_dir = dir_path / "site-packages"
        break

assert site_dir is not None, "Missing site-packages directory"
site_dir = Path(site_dir)

# Need to specially handle these snowflakes
webrtcvad_path = list(site_dir.glob("_webrtcvad.*.so"))[0]

a = Analysis(
    [Path.cwd() / "__main__.py"],
    pathex=["."],
    binaries=[(webrtcvad_path, ".")],
    datas=copy_metadata("webrtcvad"),
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="rhasspyasr_pocketsphinx_hermes",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="rhasspyasr_pocketsphinx_hermes",
)
