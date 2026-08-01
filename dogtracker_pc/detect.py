"""YOLOv8s dog detection over discovered frames, with on-disk caching.

Inference is the expensive step, so results are cached per source folder in
``<folder>/.dogtracker_cache/detections.json`` keyed by each frame's file size
and mtime. Re-running against the same folder only re-detects frames that are
new or changed.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

from PIL import Image

from .frames import Frame

logger = logging.getLogger(__name__)

# COCO class id for "dog" (the pretrained yolov8s.pt label set).
DOG_CLASS_ID = 16

CACHE_DIRNAME = ".dogtracker_cache"
CACHE_FILENAME = "detections.json"
CACHE_VERSION = 1


@dataclass(frozen=True)
class Detection:
    """A single dog detection (highest-confidence box) in one frame."""

    filename: str
    timestamp_ms: int
    frame_width: int
    frame_height: int
    x: float
    y: float
    w: float
    h: float
    confidence: float


class ProgressCallback(Protocol):
    def __call__(self, done: int, total: int) -> None: ...


def _cache_path(folder: Path) -> Path:
    return folder / CACHE_DIRNAME / CACHE_FILENAME


def _fingerprint(frame: Frame) -> str:
    return f"{frame.size}:{int(frame.mtime)}"


def load_cache(folder: Path, rotate_degrees: int = 0) -> dict:
    path = _cache_path(folder)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable detection cache: %s", exc)
        return {}
    # A cache built with a different rotation setting has box coordinates in
    # a different coordinate space -- reusing it would silently mix them up.
    if data.get("version") != CACHE_VERSION or data.get("rotate_degrees", 0) != rotate_degrees:
        return {}
    return data.get("entries", {})


def save_cache(folder: Path, entries: dict, rotate_degrees: int = 0) -> None:
    path = _cache_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": CACHE_VERSION, "rotate_degrees": rotate_degrees, "entries": entries}))
    tmp.replace(path)


def _bundled_weights_path() -> Optional[Path]:
    """Locate weights bundled next to a frozen (PyInstaller onefile) executable.

    A onefile build extracts its ``datas`` into a temp dir at ``sys._MEIPASS``
    on each launch, unrelated to the process's current working directory --
    a bare relative "yolov8s.pt" would not be found there. In a normal
    (non-frozen) run this returns None and ultralytics resolves/downloads the
    weights itself, exactly as it does during development.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    candidate = Path(meipass) / "yolov8s.pt"
    return candidate if candidate.exists() else None


def default_model_factory():
    """Lazily import ultralytics so the rest of the package works without it installed."""
    from ultralytics import YOLO

    bundled = _bundled_weights_path()
    return YOLO(str(bundled) if bundled else "yolov8s.pt")


def run_detection(
    folder: Path,
    frames: Iterable[Frame],
    model=None,
    model_factory: Callable[[], object] = default_model_factory,
    progress_cb: Optional[ProgressCallback] = None,
    use_cache: bool = True,
    rotate_degrees: int = 0,
) -> list[Detection]:
    """Run dog detection over ``frames``, reusing cached results where possible.

    ``model`` can be injected directly (e.g. in tests, or to reuse a
    already-loaded model across runs); otherwise ``model_factory`` is called
    once, lazily, only if there is at least one frame that needs detecting.

    ``rotate_degrees`` (0/90/180/270) corrects a physically rotated camera
    mount: frames are rotated clockwise by this amount before detection, and
    the resulting box coordinates (and frame_width/frame_height, swapped for
    90/270) are in that rotated space -- server.py rotates frames the same
    way when serving them, so everything lines up consistently.
    """
    folder = Path(folder)
    frames = list(frames)
    cache = load_cache(folder, rotate_degrees) if use_cache else {}
    detections: list[Detection] = []
    to_run: list[Frame] = []

    for frame in frames:
        fingerprint = _fingerprint(frame)
        cached = cache.get(frame.filename)
        if cached is not None and cached.get("fingerprint") == fingerprint:
            det = cached.get("detection")
            if det:
                detections.append(Detection(**det))
            continue
        to_run.append(frame)

    if to_run:
        if model is None:
            model = model_factory()
        total = len(to_run)
        for done, frame in enumerate(to_run, start=1):
            det = _detect_single(model, frame, rotate_degrees)
            cache[frame.filename] = {
                "fingerprint": _fingerprint(frame),
                "detection": asdict(det) if det else None,
            }
            if det:
                detections.append(det)
            if progress_cb:
                progress_cb(done, total)
        if use_cache:
            save_cache(folder, cache, rotate_degrees)

    detections.sort(key=lambda d: d.timestamp_ms)
    return detections


def _detect_single(model, frame: Frame, rotate_degrees: int = 0) -> Optional[Detection]:
    if rotate_degrees:
        with Image.open(frame.path) as img:
            source = img.convert("RGB").rotate(-rotate_degrees, expand=True)
        width, height = source.size
        results = model.predict(source=source, classes=[DOG_CLASS_ID], verbose=False)
    else:
        width, height = frame.width, frame.height
        results = model.predict(source=str(frame.path), classes=[DOG_CLASS_ID], verbose=False)

    if not results:
        return None
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    confidences = boxes.conf
    best_idx = int(confidences.argmax())
    x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[best_idx].tolist()]
    return Detection(
        filename=frame.filename,
        timestamp_ms=frame.timestamp_ms,
        frame_width=width,
        frame_height=height,
        x=(x1 + x2) / 2,
        y=(y1 + y2) / 2,
        w=x2 - x1,
        h=y2 - y1,
        confidence=float(confidences[best_idx]),
    )
