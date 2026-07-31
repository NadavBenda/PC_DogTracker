# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the DogTracker offline analysis tool.

Build on Windows (PyInstaller does not cross-compile):

    pip install -r requirements.txt pyinstaller pyinstaller-hooks-contrib
    python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
    pyinstaller packaging/dogtracker.spec --distpath dist --workpath build

See packaging/build_exe.bat, which runs all of the above in order.
"""

import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(SPEC)), ".."))
weights_path = os.path.join(repo_root, "yolov8s.pt")

if not os.path.exists(weights_path):
    raise SystemExit(
        "yolov8s.pt not found next to the repo root.\n"
        "Run: python -c \"from ultralytics import YOLO; YOLO('yolov8s.pt')\" first "
        "(see packaging/build_exe.bat)."
    )

datas = [
    (os.path.join(repo_root, "dogtracker_pc", "static"), os.path.join("dogtracker_pc", "static")),
    (weights_path, "."),
]

a = Analysis(
    [os.path.join(repo_root, "run_dogtracker.py")],
    pathex=[repo_root],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="DogTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
