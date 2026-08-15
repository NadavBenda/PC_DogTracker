"""Standalone multi-dog heatmap tool -- for a one-off recording with more
than one dog in frame at once.

The main dashboard (dogtracker_pc/) assumes a single dog: detection keeps
only the highest-confidence box per frame (see detect._detect_single), and
everything downstream -- visits, areas, the movement-path map -- is built on
that one continuous position stream. With two dogs in the same footage, that
collapses into a single fake trajectory that jumps between two unrelated
animals whenever the "winning" box switches -- not just incomplete, actively
misleading.

This script sidesteps the identity problem instead of solving it: it keeps
EVERY dog detected in each frame (not just the best one), with no attempt to
track which box is which dog across frames, and plots all of them together
as a single density heatmap -- "where dogs were", not "where each dog was".
That's exactly what a merged heatmap needs (analysis.build_heatmap already
just accumulates points; it doesn't care how many came from one frame or
which animal they belonged to), so nothing in the main pipeline
(detect.run_detection, server.py, the dashboard) is touched or has to change
to support this -- this is a fully separate entry point that only imports
already-public pieces of dogtracker_pc.

Usage:
    python two_dog_heatmap.py [frames_folder] [--rotate {90,180,270}]
                               [--blur N] [--rescan] [--output FILE.png]

With no frames_folder, a folder-picker dialog opens (same as
run_dogtracker.py). Detections are cached per folder in
.dogtracker_cache/multi_dog_detections.json -- a separate file from the
single-dog cache the main dashboard uses, so the two never collide.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from dogtracker_pc.analysis import DEFAULT_BLUR_RADIUS_PX, build_heatmap
from dogtracker_pc.detect import DOG_CLASS_ID, MIN_DETECTION_CONFIDENCE, Detection, default_model_factory
from dogtracker_pc.frames import Frame, discover_frames

logger = logging.getLogger("two_dog_heatmap")

CACHE_DIRNAME = ".dogtracker_cache"
CACHE_FILENAME = "multi_dog_detections.json"
CACHE_VERSION = 1


# ======================================================
# Detection: every box above the confidence floor, not just the best one.
# Deliberately not reusing detect._detect_single, which discards everything
# but the top box by design -- that's correct for the single-dog pipeline
# and wrong here.
# ======================================================
def _fingerprint(frame: Frame) -> str:
    return f"{frame.size}:{int(frame.mtime)}"


def _cache_path(folder: Path) -> Path:
    return folder / CACHE_DIRNAME / CACHE_FILENAME


def _load_cache(folder: Path, rotate_degrees: int) -> dict:
    path = _cache_path(folder)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable multi-dog cache: %s", exc)
        return {}
    if data.get("version") != CACHE_VERSION or data.get("rotate_degrees", 0) != rotate_degrees:
        return {}
    return data.get("entries", {})


def _save_cache(folder: Path, entries: dict, rotate_degrees: int) -> None:
    path = _cache_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": CACHE_VERSION, "rotate_degrees": rotate_degrees, "entries": entries}))
    tmp.replace(path)


def detect_all_dogs(model, frame: Frame, rotate_degrees: int = 0) -> list[Detection]:
    """Every dog box in ``frame`` at/above MIN_DETECTION_CONFIDENCE (0, 1, or more)."""
    if rotate_degrees:
        with Image.open(frame.path) as img:
            source = img.convert("RGB").rotate(-rotate_degrees, expand=True)
        width, height = source.size
        results = model.predict(source=source, classes=[DOG_CLASS_ID], verbose=False, conf=MIN_DETECTION_CONFIDENCE)
    else:
        width, height = frame.width, frame.height
        results = model.predict(
            source=str(frame.path), classes=[DOG_CLASS_ID], verbose=False, conf=MIN_DETECTION_CONFIDENCE
        )

    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
        detections.append(
            Detection(
                filename=frame.filename,
                timestamp_ms=frame.timestamp_ms,
                frame_width=width,
                frame_height=height,
                x=(x1 + x2) / 2,
                y=(y1 + y2) / 2,
                w=x2 - x1,
                h=y2 - y1,
                confidence=float(boxes.conf[i]),
            )
        )
    return detections


def run_multi_dog_detection(
    folder: Path,
    frames: list[Frame],
    model,
    rotate_degrees: int = 0,
    use_cache: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list[Detection]:
    """detect_all_dogs() over every frame, reusing cached per-frame results."""
    folder = Path(folder)
    cache = _load_cache(folder, rotate_degrees) if use_cache else {}
    all_detections: list[Detection] = []
    to_run: list[Frame] = []

    for frame in frames:
        fingerprint = _fingerprint(frame)
        cached = cache.get(frame.filename)
        if cached is not None and cached.get("fingerprint") == fingerprint:
            all_detections.extend(Detection(**d) for d in cached.get("detections", []))
            continue
        to_run.append(frame)

    total = len(to_run)
    for done, frame in enumerate(to_run, start=1):
        dets = detect_all_dogs(model, frame, rotate_degrees)
        cache[frame.filename] = {
            "fingerprint": _fingerprint(frame),
            "detections": [asdict(d) for d in dets],
        }
        all_detections.extend(dets)
        if progress_cb:
            progress_cb(done, total)

    if use_cache and to_run:
        _save_cache(folder, cache, rotate_degrees)

    all_detections.sort(key=lambda d: d.timestamp_ms)
    return all_detections


def frames_with_multiple_dogs(detections: list[Detection]) -> int:
    """How many distinct frames had 2+ dogs detected at once."""
    counts = Counter(d.filename for d in detections)
    return sum(1 for c in counts.values() if c >= 2)


# ======================================================
# Rendering: reference frame dimmed to 60% (matching the main dashboard's
# heatmap-over-reference-frame treatment) with the merged heatmap composited
# on top, flattened to a single opaque PNG.
# ======================================================
def render_heatmap_image(
    frames: list[Frame],
    detections: list[Detection],
    rotate_degrees: int = 0,
    blur_radius: int = DEFAULT_BLUR_RADIUS_PX,
) -> Image.Image:
    if not frames:
        raise ValueError("No frames to render a reference image from")

    reference_frame = frames[len(frames) // 2]
    frame_width, frame_height = reference_frame.width, reference_frame.height
    if rotate_degrees in (90, 270):
        frame_width, frame_height = frame_height, frame_width

    with Image.open(reference_frame.path) as img:
        base = img.convert("RGB")
        if rotate_degrees:
            base = base.rotate(-rotate_degrees, expand=True)

    dimmed = Image.blend(Image.new("RGB", base.size, (0, 0, 0)), base, 0.6)
    heatmap = build_heatmap(detections, frame_width, frame_height, blur_radius=blur_radius)

    composed = dimmed.convert("RGBA")
    composed.alpha_composite(heatmap)
    return composed.convert("RGB")


def _prompt_for_folder() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title="Select the folder of JPEG frames from the SD card")
        root.destroy()
    except Exception:
        logger.debug("Tk folder picker unavailable", exc_info=True)
        return None
    return Path(chosen) if chosen else None


def _console_progress(done: int, total: int) -> None:
    pct = (done / total * 100) if total else 100.0
    sys.stdout.write(f"\rDetecting dogs in frames... {done}/{total} ({pct:0.1f}%)")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-off tool: a single heatmap of every dog detected in each frame "
            "(no per-dog identity/tracking), for a recording with more than one dog."
        )
    )
    parser.add_argument("folder", nargs="?", type=Path, help="Folder of JPEG frames to analyze")
    parser.add_argument(
        "--output", type=Path, default=None, help="Output PNG path (default: <folder>/multi_dog_heatmap.png)"
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[90, 180, 270],
        default=0,
        help="Rotate frames clockwise by this many degrees before detection/rendering "
        "(use if the camera is mounted rotated, e.g. --rotate 180 for upside-down)",
    )
    parser.add_argument(
        "--blur", type=int, default=DEFAULT_BLUR_RADIUS_PX, help="Heatmap smoothing radius in pixels (default: %(default)s)"
    )
    parser.add_argument("--rescan", action="store_true", help="Ignore cached detections and re-run YOLO on every frame")
    parser.add_argument("--no-gui", action="store_true", help="Never use a Tk folder-picker dialog; FOLDER must be given")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    folder = args.folder
    if folder is None and not args.no_gui:
        folder = _prompt_for_folder()
    if folder is None:
        print("No folder given and no folder was selected. Pass a folder path as an argument.")
        return 2

    folder = Path(folder)
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 2

    print(f"Scanning {folder} for JPEG frames...")
    frames = discover_frames(folder)
    print(f"Found {len(frames)} frames")
    if not frames:
        print("No frames found -- nothing to do.")
        return 1

    print("Loading yolov8s.pt (downloads once if not already cached locally)...")
    model = default_model_factory()

    detections = run_multi_dog_detection(
        folder,
        frames,
        model,
        rotate_degrees=args.rotate,
        use_cache=not args.rescan,
        progress_cb=_console_progress,
    )

    multi_dog_frames = frames_with_multiple_dogs(detections)
    print(
        f"{len(detections)} dog detections across {len(frames)} frames "
        f"({multi_dog_frames} frame(s) had 2+ dogs at once)."
    )
    if not detections:
        print("No dogs detected at all -- the heatmap would be empty. Nothing saved.")
        return 1

    image = render_heatmap_image(frames, detections, rotate_degrees=args.rotate, blur_radius=args.blur)
    output_path = args.output or (folder / "multi_dog_heatmap.png")
    image.save(output_path)
    print(f"Saved heatmap to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
