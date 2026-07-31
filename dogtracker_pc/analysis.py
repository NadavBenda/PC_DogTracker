"""Heatmap rendering and dwell/visit segmentation over dog detections.

The heatmap uses a single-hue sequential ramp (light -> dark blue, the same
ramp used elsewhere in the UI) so magnitude reads as "more saturated / more
opaque", never as a multi-hue rainbow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from .detect import Detection

DEFAULT_DISTANCE_THRESHOLD_PX = 40.0
DEFAULT_TIME_GAP_THRESHOLD_MS = 3000
DEFAULT_BLUR_RADIUS_PX = 15


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


def _build_colormap(max_alpha: int = 235) -> np.ndarray:
    stops = np.array([_hex_to_rgb(h) for h in _BLUE_RAMP_HEX], dtype=np.float64)
    stop_positions = np.linspace(0.0, 1.0, len(stops))
    sample_positions = np.linspace(0.0, 1.0, 256)

    lut = np.zeros((256, 4), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.interp(sample_positions, stop_positions, stops[:, channel]).astype(np.uint8)
    # Alpha uses sqrt easing so low-density spots (a single visit) stay
    # visible instead of vanishing next to a much busier hotspot.
    lut[:, 3] = (np.sqrt(sample_positions) * max_alpha).astype(np.uint8)
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
        grid[yi, xi] += 1.0

    if grid.max() > 0 and blur_radius > 0:
        grid = _blur(grid, blur_radius)

    max_val = grid.max()
    if max_val <= 0:
        normalized = np.zeros((height, width), dtype=np.uint8)
    else:
        normalized = np.clip(grid / max_val * 255.0, 0, 255).astype(np.uint8)

    rgba = _COLORMAP[normalized]
    return Image.fromarray(rgba, mode="RGBA")
