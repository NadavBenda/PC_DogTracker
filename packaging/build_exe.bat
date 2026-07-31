@echo off
setlocal enabledelayedexpansion

rem Builds DogTracker.exe on Windows. Run from the repo root or from
rem packaging\ -- this script cd's to the repo root itself.
cd /d "%~dp0\.."

echo === Installing build dependencies ===
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt || goto :error
python -m pip install pyinstaller pyinstaller-hooks-contrib || goto :error

echo === Downloading YOLOv8s weights (bundled into the exe, no internet needed at runtime) ===
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" || goto :error

echo === Running PyInstaller ===
pyinstaller packaging\dogtracker.spec --distpath dist --workpath build --noconfirm || goto :error

echo.
echo Build complete: dist\DogTracker.exe
exit /b 0

:error
echo.
echo Build failed -- see the error above.
exit /b 1
