"""Heatmap rendering and dwell/visit segmentation over dog detections.

The heatmap uses a single-hue sequential ramp (light -> dark blue, the same
ramp used elsewhere in the UI) so magnitude reads as "more saturated / more
opaque", never as a multi-hue rainbow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from .detect import Detection

DEFAULT_DISTANCE_THRESHOLD_PX = 40.0
DEFAULT_TIME_GAP_THRESHOLD_MS = 3000
DEFAULT_BLUR_RADIUS_PX = 15
DEFAULT_AREA_RADIUS_PX = 60.0
DEFAULT_TOP_AREAS_COUNT = 3
# Convex hulls are padded outward by this many pixels so the polygon visibly
# encloses the detections that produced it instead of hugging their exact
# pixels.
AREA_HULL_PAD_PX = 14.0


@dataclass(frozen=True)
class Visit:
    """A contiguous dwell in roughly one spot before the dog moved on or left."""

    start_ts: int
    end_ts: int
    duration_ms: int
    centroid_x: float
    centroid_y: float
    frame_count: int
    detection_indices: tuple[int, ...]


def segment_visits(
    detections: Sequence[Detection],
    distance_threshold_px: float = DEFAULT_DISTANCE_THRESHOLD_PX,
    time_gap_threshold_ms: int = DEFAULT_TIME_GAP_THRESHOLD_MS,
) -> list[Visit]:
    """Group time-ordered detections into visits.

    A visit continues while consecutive detections stay within
    ``distance_threshold_px`` of each other and land within
    ``time_gap_threshold_ms``; either the dog moving away or a gap in
    detections (it left the frame, or a sample was missed) closes the visit.
    """
    if not detections:
        return []

    def flush(indices: list[int]) -> Visit:
        ts = [detections[i].timestamp_ms for i in indices]
        xs = [detections[i].x for i in indices]
        ys = [detections[i].y for i in indices]
        return Visit(
            start_ts=min(ts),
            end_ts=max(ts),
            duration_ms=max(ts) - min(ts),
            centroid_x=sum(xs) / len(xs),
            centroid_y=sum(ys) / len(ys),
            frame_count=len(indices),
            detection_indices=tuple(indices),
        )

    visits: list[Visit] = []
    current: list[int] = [0]

    for i in range(1, len(detections)):
        prev, curr = detections[i - 1], detections[i]
        dist = ((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2) ** 0.5
        gap = curr.timestamp_ms - prev.timestamp_ms
        if dist <= distance_threshold_px and gap <= time_gap_threshold_ms:
            current.append(i)
        else:
            visits.append(flush(current))
            current = [i]

    visits.append(flush(current))
    return visits


@dataclass(frozen=True)
class Area:
    """Several visits close enough together to count as "the same small spot"."""

    centroid_x: float
    centroid_y: float
    visit_count: int
    total_duration_ms: int
    avg_duration_ms: float
    visit_indices: tuple[int, ...]


def find_areas(visits: Sequence[Visit], area_radius_px: float = DEFAULT_AREA_RADIUS_PX) -> list[Area]:
    """Group visits into small areas by proximity, ranked by how often they recur.

    This is a second, coarser pass over the same visits `segment_visits`
    already produced: two visits to (roughly) the same spot, separated by a
    trip elsewhere, become one "area" with a visit count and an average
    dwell time -- "the dog came back to the doorway 5 times, ~1 min each".

    A simple greedy single-link clustering (each visit joins the nearest
    existing area within ``area_radius_px``, else starts a new one) is used
    rather than a general clustering algorithm: the input is a handful of
    already-aggregated visits, not raw detections, so an exact/optimal
    clustering isn't worth the extra complexity.
    """
    if not visits:
        return []

    clusters: list[dict] = []
    for i, visit in enumerate(visits):
        best_cluster = None
        best_dist = None
        for cluster in clusters:
            dist = ((visit.centroid_x - cluster["cx"]) ** 2 + (visit.centroid_y - cluster["cy"]) ** 2) ** 0.5
            if dist <= area_radius_px and (best_dist is None or dist < best_dist):
                best_cluster = cluster
                best_dist = dist

        weight = max(visit.duration_ms, 1)
        if best_cluster is None:
            clusters.append(
                {
                    "cx": visit.centroid_x,
                    "cy": visit.centroid_y,
                    "weight": weight,
                    "total_duration_ms": visit.duration_ms,
                    "members": [i],
                }
            )
        else:
            prior_weight = best_cluster["weight"]
            best_cluster["cx"] = (best_cluster["cx"] * prior_weight + visit.centroid_x * weight) / (
                prior_weight + weight
            )
            best_cluster["cy"] = (best_cluster["cy"] * prior_weight + visit.centroid_y * weight) / (
                prior_weight + weight
            )
            best_cluster["weight"] = prior_weight + weight
            best_cluster["total_duration_ms"] += visit.duration_ms
            best_cluster["members"].append(i)

    areas = [
        Area(
            centroid_x=c["cx"],
            centroid_y=c["cy"],
            visit_count=len(c["members"]),
            total_duration_ms=c["total_duration_ms"],
            avg_duration_ms=c["total_duration_ms"] / len(c["members"]),
            visit_indices=tuple(c["members"]),
        )
        for c in clusters
    ]
    areas.sort(key=lambda a: (a.visit_count, a.total_duration_ms), reverse=True)
    return areas


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone-chain convex hull. Returns points in CCW order."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def hull_polygon(
    points: Sequence[tuple[float, float]], pad_px: float = AREA_HULL_PAD_PX
) -> list[tuple[float, float]]:
    """Convex hull of ``points``, padded outward so it visibly encloses them
    instead of hugging the exact pixels they were detected at.

    Degenerate input (a single point, or points that are all collinear so the
    "hull" is just a line) falls back to a small polygon centered on the
    points so there is still a visible shape to draw.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if not pts:
        return []

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    hull = _convex_hull(pts)
    if len(hull) < 3:
        spread = max((((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in pts), default=0.0)
        radius = max(spread, pad_px)
        steps = 8
        return [
            (
                cx + radius * math.cos(2 * math.pi * k / steps),
                cy + radius * math.sin(2 * math.pi * k / steps),
            )
            for k in range(steps)
        ]

    padded = []
    for x, y in hull:
        dx, dy = x - cx, y - cy
        dist = (dx**2 + dy**2) ** 0.5
        if dist < 1e-6:
            padded.append((x, y))
        else:
            scale = (dist + pad_px) / dist
            padded.append((cx + dx * scale, cy + dy * scale))
    return padded


# Sequential blue ramp, steps 100->700 from the shared reference palette
# (light/near-surface -> dark/saturated). Position is normalized 0..1 across
# the named steps so the LUT below can interpolate a smooth 256-entry ramp.
_BLUE_RAMP_HEX = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _build_colormap(max_alpha: int = 255) -> np.ndarray:
    stops = np.array([_hex_to_rgb(h) for h in _BLUE_RAMP_HEX], dtype=np.float64)
    stop_positions = np.linspace(0.0, 1.0, len(stops))
    sample_positions = np.linspace(0.0, 1.0, 256)

    lut = np.zeros((256, 4), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.interp(sample_positions, stop_positions, stops[:, channel]).astype(np.uint8)
    # Alpha uses aggressive easing (< sqrt) so even lightly-visited spots read
    # as solid and dark rather than a faint wash.
    lut[:, 3] = (sample_positions**0.4 * max_alpha).astype(np.uint8)
    return lut


_COLORMAP = _build_colormap()


def _gaussian_kernel1d(radius: int) -> np.ndarray:
    sigma = max(radius / 3.0, 1e-6)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _blur(grid: np.ndarray, radius: int) -> np.ndarray:
    """Separable Gaussian blur, implemented directly on the float density grid.

    PIL's ``ImageFilter.GaussianBlur`` refuses 32-bit-float ("F") images, and
    converting to 8-bit first would clip the density values before they are
    normalized, so the kernel is applied by hand instead.
    """
    if radius <= 0:
        return grid
    kernel = _gaussian_kernel1d(int(radius))
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=grid)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=blurred)
    return blurred


# Each detection is stamped as a small filled disc rather than a single
# pixel, so individual visits already read as a visible dot before the
# smoothing blur is applied on top.
_POINT_SPLAT_RADIUS_PX = 5


def _stamp_point(grid: np.ndarray, xi: int, yi: int, radius: int) -> None:
    height, width = grid.shape
    y0, y1 = max(0, yi - radius), min(height, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(width, xi + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - xi) ** 2 + (yy - yi) ** 2 <= radius * radius
    grid[y0:y1, x0:x1][mask] += 1.0


def build_heatmap(
    detections: Sequence[Detection],
    width: int,
    height: int,
    blur_radius: int = DEFAULT_BLUR_RADIUS_PX,
) -> Image.Image:
    """Render an RGBA density heatmap of detection centers, sized ``width`` x ``height``."""
    width = max(int(width), 1)
    height = max(int(height), 1)
    grid = np.zeros((height, width), dtype=np.float64)

    for det in detections:
        xi = int(min(max(det.x, 0), width - 1))
        yi = int(min(max(det.y, 0), height - 1))
        _stamp_point(grid, xi, yi, _POINT_SPLAT_RADIUS_PX)

    if grid.max() > 0 and blur_radius > 0:
        grid = _blur(grid, blur_radius)

    max_val = grid.max()
    if max_val <= 0:
        normalized = np.zeros((height, width), dtype=np.uint8)
    else:
        normalized = np.clip(grid / max_val * 255.0, 0, 255).astype(np.uint8)

    rgba = _COLORMAP[normalized]
    return Image.fromarray(rgba, mode="RGBA")
