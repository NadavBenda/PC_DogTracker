"""Manually-drawn preferred-location regions, persisted per source folder.

Complements the automatically-clustered areas in analysis.py: the user can
trace an arbitrary polygon directly on the heatmap to mark a spot the
automatic clustering missed or split awkwardly. Stored in the same per-folder
cache directory as the detection cache, so it survives reopening the same
session -- same on-disk pattern as detect.py's cache (versioned JSON,
atomic write via a temp file + replace).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

CACHE_DIRNAME = ".dogtracker_cache"
CACHE_FILENAME = "manual_regions.json"
CACHE_VERSION = 1


@dataclass(frozen=True)
class ManualRegion:
    """A user-drawn polygon, in frame-pixel coordinates."""

    id: str
    points: tuple[tuple[float, float], ...]


def _cache_path(folder: Path) -> Path:
    return folder / CACHE_DIRNAME / CACHE_FILENAME


def load_manual_regions(folder: Path) -> list[ManualRegion]:
    path = _cache_path(folder)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable manual regions cache: %s", exc)
        return []
    if data.get("version") != CACHE_VERSION:
        return []
    return [
        ManualRegion(id=r["id"], points=tuple((float(x), float(y)) for x, y in r["points"]))
        for r in data.get("regions", [])
    ]


def save_manual_regions(folder: Path, regions: Sequence[ManualRegion]) -> None:
    path = _cache_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "regions": [{"id": r.id, "points": [list(p) for p in r.points]} for r in regions],
            }
        )
    )
    tmp.replace(path)


def add_manual_region(folder: Path, points: Sequence[tuple[float, float]]) -> ManualRegion:
    regions = load_manual_regions(folder)
    region = ManualRegion(
        id=uuid.uuid4().hex[:8],
        points=tuple((float(x), float(y)) for x, y in points),
    )
    regions.append(region)
    save_manual_regions(folder, regions)
    return region


def delete_manual_region(folder: Path, region_id: str) -> bool:
    """Remove a region by id. Returns False if no region had that id."""
    regions = load_manual_regions(folder)
    remaining = [r for r in regions if r.id != region_id]
    if len(remaining) == len(regions):
        return False
    save_manual_regions(folder, remaining)
    return True


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test (even-odd rule)."""
    if len(polygon) < 3:
        return False
    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside
