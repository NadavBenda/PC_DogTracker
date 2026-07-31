"""YOLOv8s dog detection over discovered frames, with on-disk caching.

Inference is the expensive step, so results are cached per source folder in
``<folder>/.dogtracker_cache/detections.json`` keyed by each frame's file size
and mtime. Re-running against the same folder only re-detects frames that are
new or changed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

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


def load_cache(folder: Path) -> dict:
    path = _cache_path(folder)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable detection cache: %s", exc)
        return {}
    if data.get("version") != CACHE_VERSION:
        return {}
    return data.get("entries", {})


def save_cache(folder: Path, entries: dict) -> None:
    path = _cache_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": CACHE_VERSION, "entries": entries}))
    tmp.replace(path)


def default_model_factory():
    """Lazily import ultralytics so the rest of the package works without it installed."""
    from ultralytics import YOLO

    return YOLO("yolov8s.pt")


def run_detection(
    folder: Path,
    frames: Iterable[Frame],
    model=None,
    model_factory: Callable[[], object] = default_model_factory,
    progress_cb: Optional[ProgressCallback] = None,
    use_cache: bool = True,
) -> list[Detection]:
    """Run dog detection over ``frames``, reusing cached results where possible.

    ``model`` can be injected directly (e.g. in tests, or to reuse a
    already-loaded model across runs); otherwise ``model_factory`` is called
    once, lazily, only if there is at least one frame that needs detecting.
    """
    folder = Path(folder)
    frames = list(frames)
    cache = load_cache(folder) if use_cache else {}
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
            det = _detect_single(model, frame)
            cache[frame.filename] = {
                "fingerprint": _fingerprint(frame),
                "detection": asdict(det) if det else None,
            }
            if det:
                detections.append(det)
            if progress_cb:
                progress_cb(done, total)
        if use_cache:
            save_cache(folder, cache)

    detections.sort(key=lambda d: d.timestamp_ms)
    return detections


def _detect_single(model, frame: Frame) -> Optional[Detection]:
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
        frame_width=frame.width,
        frame_height=frame.height,
        x=(x1 + x2) / 2,
        y=(y1 + y2) / 2,
        w=x2 - x1,
        h=y2 - y1,
        confidence=float(confidences[best_idx]),
    )
