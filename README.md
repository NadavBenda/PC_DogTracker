# PC_DogTracker

Offline, one-click Windows tool for the ESP32-S3 Dog Trajectory Tracker project.

Takes the JPEG frames recorded to the ESP32's SD card and runs YOLOv8s object
detection over all of them locally (no internet required), then presents an
interactive heatmap + dwell/visit analysis with click-to-jump frame browsing,
served as a local web UI and packaged as a single `.exe`.

![Dashboard overview](docs/screenshots/dashboard.png)

*(Screenshot uses a synthetic demo yard/doghouse scene, not real camera
footage -- generated for illustration purposes, see below.)*

## What you get

- **Visit heatmap** overlaid on a frame from the middle of the session (a
  more representative "what does this scene normally look like" reference
  than frame 0); pick any other frame as the background with one click.
- **Most visited spot** -- the small area the dog returned to most often,
  with a count, average/total dwell time, and one thumbnail per visit.
- **Frame browser** with the detected bounding box drawn on it, a scrubber,
  and prev/next controls.
- **Click-to-jump** everywhere: click a point on the heatmap, a row in the
  visits table, or a thumbnail in the most-visited-spot strip, and the frame
  browser jumps straight to it.
- **Adjustable thresholds** (dwell radius, gap tolerance, area radius,
  heatmap smoothing) recompute the visits/areas/heatmap live.
- Runs entirely on `127.0.0.1`; light/dark mode follow your system.

<img src="docs/screenshots/most_visited_spot.png" alt="Most visited spot card" width="700" />

## How it works

1. Scans the chosen folder for JPEG frames (`frames.py`), parsing a
   timestamp out of each filename (falls back to file mtime).
2. Runs YOLOv8s over each frame looking for the `dog` class, keeping the
   highest-confidence box per frame (`detect.py`). Results are cached per
   folder in `.dogtracker_cache/`, keyed by file size + mtime, so re-opening
   the same session skips detection entirely.
3. Groups detections into **visits** (a contiguous dwell in roughly one
   spot) and **areas** (visits close enough together to count as the same
   small spot, e.g. repeated trips to a doghouse) (`analysis.py`).
4. Serves everything from a small Flask app (`server.py` + `static/`) --
   no telemetry, no external requests, nothing leaves the machine.

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

## Firmware (ESP32-S3, `firmware/DogTracker/DogTracker.ino`)

The companion sketch that produces the SD card data this tool analyzes:

- Captures JPEG frames continuously at **5 FPS** and saves each one to the
  SD card as `/<session>/dog_<millis>.jpg` (point this tool at one session
  folder, not the SD root). Recording stops cleanly once the card fills
  up -- existing files are never overwritten or deleted.
- For the **first 2 minutes only**, opens a Wi-Fi access point named
  `WatchDog` (password `12345678`) and serves a 1 FPS MJPEG preview at
  `http://192.168.4.1/`, so you can check camera placement/focus from a
  phone before walking away. After 2 minutes the AP and HTTP server shut
  down completely (no lingering open network) and only the 5 FPS SD
  recording continues.
- **No RTC on the board**, so the session folder starts out named from
  milliseconds-since-boot. Opening `http://192.168.4.1/` during the AP
  window sends your browser's clock to the board automatically, which
  renames new frames' folder to a real `YYYYMMDD_HHMMSS` (local time --
  see `TZ_OFFSET_SECONDS` in the sketch, defaults to UTC+2). Frames
  captured before that sync stay in the original boot-relative folder
  rather than being moved.
- No on-device detection -- that's what this PC tool is for. The ESP32
  only captures and saves frames.

Verified with a real PlatformIO compile against `esp32-s3-devkitc-1`
(Arduino framework) -- 14.9% RAM / 27.8% flash used, no warnings.

**Before flashing, confirm the SD card wiring.** The sketch assumes
SD_MMC 1-bit mode on GPIO39 (CLK), GPIO38 (CMD), GPIO40 (D0) -- a common
pinout for this class of board, but unverified against your specific
board's schematic. If it's wrong, `SD_MMC.begin()` fails loudly over
Serial at boot (nothing gets damaged) -- update `SD_MMC_CLK_PIN` /
`SD_MMC_CMD_PIN` / `SD_MMC_D0_PIN` at the top of the sketch and reflash.

Arduino IDE board settings:
- Board: **ESP32S3 Dev Module**
- USB CDC On Boot: **Enabled** (for Serial over USB)
- Flash Size: **16MB**
- Partition Scheme: **Huge APP** (or similar, with room for camera/Wi-Fi/SD)
- PSRAM: **OPI PSRAM**

## Dark mode

The dashboard follows your OS's light/dark setting automatically.

<img src="docs/screenshots/dashboard_dark.png" alt="Dashboard in dark mode" width="700" />
