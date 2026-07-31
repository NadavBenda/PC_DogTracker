# PC_DogTracker

Offline, one-click Windows tool for the ESP32-S3 Dog Trajectory Tracker project.

Takes the JPEG frames recorded to the ESP32's SD card and runs YOLOv8s object
detection over all of them locally (no internet required), then presents an
interactive heatmap + dwell/visit analysis with click-to-jump frame browsing,
served as a local web UI and packaged as a single `.exe`.

## Running from source

```
pip install -r requirements.txt
python run_dogtracker.py                 # prompts for a folder via a file dialog
python run_dogtracker.py path\to\frames   # or pass it directly
```

The first run downloads the `yolov8s.pt` weights (needs internet once); after
that, detections are cached per source folder in `.dogtracker_cache/` so
re-opening the same session is instant. A browser tab opens automatically at
`http://127.0.0.1:5151/`.

Useful flags: `--rescan` (ignore the cache and re-run detection on every
frame), `--no-browser`, `--no-gui` (skip the Tk dialogs; requires passing a
folder), `--port`, `-v`/`--verbose`.

## Building DogTracker.exe (Windows only)

PyInstaller does not cross-compile, so the `.exe` must be built on Windows:

```
packaging\build_exe.bat
```

This installs the build dependencies, downloads the YOLO weights, bundles
them + the web UI assets into the executable, and writes
`dist\DogTracker.exe`. That exe needs no internet connection and no Python
install on the machine that runs it. See `packaging/dogtracker.spec` for the
PyInstaller configuration.

Because it bundles PyTorch/OpenCV, the onefile exe is large (400MB+) and
**re-extracts itself to a temp folder on every launch**, so expect roughly
10-15 seconds of "nothing happening" before the window/browser appears --
that's normal, not a hang. If that startup delay matters more than shipping
a single file, switch `packaging/dogtracker.spec`'s `EXE(...)` to PyInstaller's
`--onedir`-style `COLLECT(...)` output instead; everything else stays the
same.

## Running the tests

```
pip install pytest
pytest
```
