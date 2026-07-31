"""Local (127.0.0.1-only) Flask app serving the trajectory dashboard."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from .analysis import (
    DEFAULT_BLUR_RADIUS_PX,
    DEFAULT_DISTANCE_THRESHOLD_PX,
    DEFAULT_TIME_GAP_THRESHOLD_MS,
    build_heatmap,
    segment_visits,
)
from .detect import Detection
from .frames import Frame

logger = logging.getLogger(__name__)


def create_app(folder: Path, frames: list[Frame], detections: list[Detection]) -> Flask:
    folder = Path(folder)
    app = Flask(__name__)

    # Only files we already discovered as frames may be served back out, so a
    # crafted filename in the URL (e.g. "../../secrets.txt") can't escape the
    # source folder -- send_from_directory only ever runs against a name that
    # was already enumerated from disk.
    frame_by_name = {f.filename: f for f in frames}
    frame_width = frames[0].width if frames else 0
    frame_height = frames[0].height if frames else 0

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
                "defaults": {
                    "distance_threshold_px": DEFAULT_DISTANCE_THRESHOLD_PX,
                    "time_gap_threshold_ms": DEFAULT_TIME_GAP_THRESHOLD_MS,
                    "blur_radius_px": DEFAULT_BLUR_RADIUS_PX,
                },
            }
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
        return send_from_directory(folder, filename)

    return app
