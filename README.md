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

- **Visit heatmap** and **frame browser** side by side (each half the
  screen), overlaid on a frame from the middle of the session (a more
  representative "what does this scene normally look like" reference than
  frame 0); pick any other frame as the background with one click. The frame
  browser marks the dog's exact center point too, not just its bounding box,
  matching the point the heatmap and regions are built from.
- **Preferred locations** -- the dog's handful of regular spots (e.g. a
  doghouse, a water bowl), each outlined as its own colored **convex hull**
  directly on the heatmap image (one color per region, matching that
  region's heading below, purely visual -- it doesn't block clicking on the
  heatmap underneath). Per region: how many times it was visited, the length
  of every individual visit, the total time spent there, and its own
  clickable **presence timeline** (in the region's color) showing exactly
  when across the whole session the dog was there vs elsewhere. A summary
  line splits total tracked time into "in a highlighted region" vs.
  "elsewhere (in transit)" -- time spent moving between spots rather than
  dwelling in one of them.
- **Movement path over time** -- a smooth spline through every detected
  position in chronological order, colored gradually from one color (earlier)
  to another (later) by elapsed time, with numbered markers at every 10% of
  the session so the dog's actual route -- not just where it stopped -- is
  readable as a timeline.
- **Frame browser** scrubs through every recorded frame, not just the ones
  with a detection -- frames with no dog show "No detection" instead of a
  box, so you can review the full sequence for continuity. A clickable
  **detection-presence ruler** below the scrubber shows at a glance which
  parts of the session had a dog detected (colored) vs not (empty).
- **Click-to-jump** everywhere: click a point on the heatmap, a row in the
  visits table, a region's presence timeline, or a thumbnail in a region's
  visit strip, and the frame browser jumps straight to it.
- **Adjustable thresholds** (dwell radius, gap tolerance, area radius,
  heatmap smoothing, how many regions to highlight) recompute the
  visits/regions/heatmap live.
- Runs entirely on `127.0.0.1`; light/dark mode follow your system.

<img src="docs/screenshots/preferred_locations.png" alt="Preferred locations card" width="700" />

## How it works

1. Scans the chosen folder for JPEG frames (`frames.py`), parsing a
   timestamp out of each filename (falls back to file mtime).
2. Runs YOLOv8s over each frame looking for the `dog` class, keeping the
   highest-confidence box per frame (`detect.py`). Results are cached per
   folder in `.dogtracker_cache/`, keyed by file size + mtime, so re-opening
   the same session skips detection entirely.
3. Groups detections into **visits** (a contiguous dwell in roughly one
   spot) and **areas** (visits close enough together to count as the same
   small spot, e.g. repeated trips to a doghouse); the top few areas by
   visit count/duration are highlighted as the dog's "preferred locations",
   with everything else counted as time elsewhere (`analysis.py`).
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
frame), `--rotate {90,180,270}` (if the camera is mounted rotated -- corrects
both detection and the images shown in the dashboard; changing this
invalidates the detection cache since box coordinates are in a different
coordinate space), `--no-browser`, `--no-gui` (skip the Tk dialogs; requires
passing a folder), `--port`, `-v`/`--verbose`.

If detection comes back empty on real footage, `python debug_detect.py
photo1.jpg photo2.jpg` (repo root) runs YOLOv8s directly on 1-3 images and
prints every candidate object at any confidence (not just "dog"), plus saves
an annotated copy -- it quickly tells you whether the model is missing the
dog completely (focus/distance/lighting) or seeing it "almost" at a
confidence just under the usual cutoff.

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

- Captures JPEG frames continuously and saves each one to the SD card as
  `/<session>/dog_<millis>.jpg` (point this tool at one session folder,
  not the SD root). Baseline rate is **3 FPS**; it drops to **2 FPS**
  while the network window (below) is open, so streaming doesn't have to
  contend with SD writes for CPU/bandwidth. Recording stops cleanly once
  the card fills up -- existing files are never overwritten or deleted.
- At boot, tries to join one of a list of known home Wi-Fi networks (in
  order, ~8s each) for up to **4 minutes**. If one connects: gets real
  time via **NTP** automatically, and serves a live MJPEG preview *and* a
  client-side **focus-score readout** at `http://<device-ip>/` -- a
  live-updating sharpness number (computed in the browser, no server load)
  so you can watch it while turning the lens's physical focus ring. After
  the window (or immediately, if no known network was reachable), Wi-Fi
  shuts down completely and only SD recording continues at the full rate.
- **No RTC on the board.** If NTP synced, the session folder is renamed to
  a real `YYYYMMDD_HHMMSS` (local time -- see `TZ_OFFSET_SECONDS` in the
  sketch, currently UTC+3 for Israel Daylight Time; flip to UTC+2 in winter);
  frames captured before that sync stay in the
  original boot-relative folder rather than being moved. If no network was
  reachable at all that boot, the folder keeps its boot-relative fallback
  name -- made collision-proof across power cycles by a boot counter
  persisted on the SD card (`millis()` alone resets every reboot and isn't
  reliably unique if boot timing is consistent, which is exactly what
  caused two different recordings to land in the same folder in testing).
- No on-device detection -- that's what this PC tool is for. The ESP32
  only captures and saves frames. If detection comes back empty on real
  footage, see `debug_detect.py` below.
- Every 30 seconds, Serial prints a status line: the active session
  folder, whether NTP time synced, frames captured so far, elapsed time,
  free SD space, and -- while the network window is open -- a countdown
  to when it closes.
- **Hold the board's BOOT button (GPIO0) for ~1.5s** at any time to stop
  recording and cleanly unmount the SD card before cutting power -- the
  status LED switches to a fast strobe once it's actually safe to unplug.
  Cutting power without doing this risks FAT corruption (same as pulling
  a USB drive without ejecting it), which is exactly what happened during
  testing on a session that was just unplugged directly.

Verified with a real PlatformIO compile against `esp32-s3-devkitc-1`
(Arduino framework) -- 15.3% RAM / 28.1% flash used, no warnings.

**Before flashing, confirm the SD card wiring.** The sketch assumes
SD_MMC 1-bit mode on GPIO39 (CLK), GPIO38 (CMD), GPIO40 (D0) -- a common
pinout for this class of board, but unverified against your specific
board's schematic. If it's wrong, `SD_MMC.begin()` fails loudly over
Serial at boot (nothing gets damaged) -- update `SD_MMC_CLK_PIN` /
`SD_MMC_CMD_PIN` / `SD_MMC_D0_PIN` at the top of the sketch and reflash.

**Wi-Fi networks are hardcoded** in `WIFI_CANDIDATES` near the top of the
sketch -- edit that list for your own network name(s)/password(s) before
flashing.

Arduino IDE board settings:
- Board: **ESP32S3 Dev Module**
- USB CDC On Boot: **Enabled** (for Serial over USB)
- Flash Size: **16MB**
- Partition Scheme: **Huge APP** (or similar, with room for camera/Wi-Fi/SD)
- PSRAM: **OPI PSRAM**

## Dark mode

The dashboard follows your OS's light/dark setting automatically.

<img src="docs/screenshots/dashboard_dark.png" alt="Dashboard in dark mode" width="700" />
