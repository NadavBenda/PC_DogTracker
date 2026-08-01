"""Local (127.0.0.1-only) Flask app serving the trajectory dashboard."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from PIL import Image

from .analysis import (
    DEFAULT_AREA_RADIUS_PX,
    DEFAULT_BLUR_RADIUS_PX,
    DEFAULT_DISTANCE_THRESHOLD_PX,
    DEFAULT_TIME_GAP_THRESHOLD_MS,
    build_heatmap,
    find_areas,
    segment_visits,
)
from .detect import Detection
from .frames import Frame

logger = logging.getLogger(__name__)


def _static_folder() -> Path:
    """Resolve the static asset folder, on disk or inside a onefile bundle.

    A PyInstaller onefile build imports this package from a zipped archive,
    so ``Path(__file__).parent`` no longer points at real files on disk --
    the bundled ``datas`` land under ``sys._MEIPASS`` instead.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "dogtracker_pc" / "static"
    return Path(__file__).resolve().parent / "static"


def create_app(
    folder: Path,
    frames: list[Frame],
    detections: list[Detection],
    rotate_degrees: int = 0,
) -> Flask:
    folder = Path(folder)
    app = Flask(__name__, static_folder=str(_static_folder()), static_url_path="/static")

    # Only files we already discovered as frames may be served back out, so a
    # crafted filename in the URL (e.g. "../../secrets.txt") can't escape the
    # source folder -- send_from_directory only ever runs against a name that
    # was already enumerated from disk.
    frame_by_name = {f.filename: f for f in frames}
    frame_width = frames[0].width if frames else 0
    frame_height = frames[0].height if frames else 0
    if rotate_degrees in (90, 270):
        frame_width, frame_height = frame_height, frame_width
    # A frame from the temporal middle of the session tends to be a more
    # representative "what does the scene normally look like" reference than
    # frame 0, which can catch camera warm-up, a passerby, or bad light.
    reference_frame = frames[len(frames) // 2].filename if frames else None

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/summary")
    def summary():
        first_ts = frames[0].timestamp_ms if frames else 0
        last_ts = frames[-1].timestamp_ms if frames else 0
        return jsonify(
            {
                "folder": str(folder),
                "frame_count": len(frames),
                "detection_count": len(detections),
                "frame_width": frame_width,
                "frame_height": frame_height,
                "first_timestamp_ms": first_ts,
                "last_timestamp_ms": last_ts,
                "duration_ms": max(last_ts - first_ts, 0),
                "reference_frame": reference_frame,
                "defaults": {
                    "distance_threshold_px": DEFAULT_DISTANCE_THRESHOLD_PX,
                    "time_gap_threshold_ms": DEFAULT_TIME_GAP_THRESHOLD_MS,
                    "blur_radius_px": DEFAULT_BLUR_RADIUS_PX,
                    "area_radius_px": DEFAULT_AREA_RADIUS_PX,
                },
            }
        )

    @app.get("/api/frames")
    def list_frames():
        return jsonify(
            [
                {"index": i, "filename": f.filename, "timestamp_ms": f.timestamp_ms}
                for i, f in enumerate(frames)
            ]
        )

    @app.get("/api/detections")
    def list_detections():
        return jsonify(
            [
                {
                    "index": i,
                    "filename": d.filename,
                    "timestamp_ms": d.timestamp_ms,
                    "x": d.x,
                    "y": d.y,
                    "w": d.w,
                    "h": d.h,
                    "confidence": d.confidence,
                }
                for i, d in enumerate(detections)
            ]
        )

    @app.get("/api/visits")
    def list_visits():
        distance = request.args.get("distance", default=DEFAULT_DISTANCE_THRESHOLD_PX, type=float)
        gap = request.args.get("gap", default=DEFAULT_TIME_GAP_THRESHOLD_MS, type=int)
        visits = segment_visits(detections, distance_threshold_px=distance, time_gap_threshold_ms=gap)
        return jsonify(
            [
                {
                    "start_ts": v.start_ts,
                    "end_ts": v.end_ts,
                    "duration_ms": v.duration_ms,
                    "centroid_x": v.centroid_x,
                    "centroid_y": v.centroid_y,
                    "frame_count": v.frame_count,
                    "first_index": v.detection_indices[0],
                    "mid_index": v.detection_indices[len(v.detection_indices) // 2],
                    "last_index": v.detection_indices[-1],
                }
                for v in visits
            ]
        )

    @app.get("/api/areas")
    def list_areas():
        distance = request.args.get("distance", default=DEFAULT_DISTANCE_THRESHOLD_PX, type=float)
        gap = request.args.get("gap", default=DEFAULT_TIME_GAP_THRESHOLD_MS, type=int)
        area_radius = request.args.get("area_radius", default=DEFAULT_AREA_RADIUS_PX, type=float)

        visits = segment_visits(detections, distance_threshold_px=distance, time_gap_threshold_ms=gap)
        areas = find_areas(visits, area_radius_px=area_radius)

        def visit_summary(visit_index: int) -> dict:
            v = visits[visit_index]
            mid = v.detection_indices[len(v.detection_indices) // 2]
            return {
                "start_ts": v.start_ts,
                "duration_ms": v.duration_ms,
                "frame_count": v.frame_count,
                "representative_filename": detections[mid].filename,
                "representative_index": mid,
            }

        return jsonify(
            [
                {
                    "rank": rank,
                    "centroid_x": area.centroid_x,
                    "centroid_y": area.centroid_y,
                    "visit_count": area.visit_count,
                    "total_duration_ms": area.total_duration_ms,
                    "avg_duration_ms": area.avg_duration_ms,
                    "visits": [visit_summary(i) for i in area.visit_indices],
                }
                for rank, area in enumerate(areas, start=1)
            ]
        )

    @app.get("/api/heatmap.png")
    def heatmap_png():
        blur = request.args.get("blur", default=DEFAULT_BLUR_RADIUS_PX, type=int)
        img = build_heatmap(detections, frame_width or 320, frame_height or 240, blur_radius=blur)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), mimetype="image/png")

    @app.get("/api/nearest")
    def nearest():
        x = request.args.get("x", type=float)
        y = request.args.get("y", type=float)
        if x is None or y is None:
            abort(400, description="x and y query params are required")
        if not detections:
            abort(404, description="No detections available")

        best_index = 0
        best_dist_sq: Optional[float] = None
        for i, d in enumerate(detections):
            dist_sq = (d.x - x) ** 2 + (d.y - y) ** 2
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_index = i
        return jsonify({"index": best_index})

    @app.get("/frames/<path:filename>")
    def serve_frame(filename: str):
        if filename not in frame_by_name:
            abort(404)
        if not rotate_degrees:
            return send_from_directory(folder, filename)

        with Image.open(folder / filename) as img:
            rotated = img.convert("RGB").rotate(-rotate_degrees, expand=True)
            buf = io.BytesIO()
            rotated.save(buf, format="JPEG", quality=90)
        return Response(buf.getvalue(), mimetype="image/jpeg")

    return app
