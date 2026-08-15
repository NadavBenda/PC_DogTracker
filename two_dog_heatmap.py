"""Standalone multi-dog heatmap tool -- for a one-off recording with more
than one dog in frame at once.

The main dashboard's default mode assumes a single dog: detection keeps
only the highest-confidence box per frame, and visits/areas/the movement-
path map are all built on that one continuous position stream. The
dashboard also has an optional "detect 2+ dogs" toggle now (see
server.py/app.js) that switches to the same multi-dog detection this script
uses, but hides the movement-path map (a spline through points from
different, untracked dogs isn't a meaningful route).

This script is for doing that same multi-dog analysis without the
dashboard at all -- just a single flattened PNG. It keeps EVERY dog
detected in each frame (not just the best one), with no attempt to track
which box is which dog across frames, and plots all of them together as a
single density heatmap -- "where dogs were", not "where each dog was".
That's exactly what a merged heatmap needs (analysis.build_heatmap already
just accumulates points; it doesn't care how many came from one frame or
which animal they belonged to). Detection itself
(detect.detect_all_dogs/run_multi_dog_detection) is shared with the
dashboard's toggle, including the cache file, so running this script first
and then switching the toggle on in the dashboard (or vice versa) reuses
the same cached results instead of re-detecting.

Usage:
    python two_dog_heatmap.py [frames_folder] [--rotate {90,180,270}]
                               [--blur N] [--rescan] [--output FILE.png]

With no frames_folder, a folder-picker dialog opens (same as
run_dogtracker.py). Detections are cached per folder in
.dogtracker_cache/multi_dog_detections.json -- a separate file from the
single-dog cache the main dashboard's default mode uses, so the two never
collide.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

from dogtracker_pc.analysis import DEFAULT_BLUR_RADIUS_PX, build_heatmap
from dogtracker_pc.detect import Detection, default_model_factory, frames_with_multiple_dogs, run_multi_dog_detection
from dogtracker_pc.frames import Frame, discover_frames

logger = logging.getLogger("two_dog_heatmap")


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
