from pathlib import Path

import pytest

from dogtracker_pc.detect import Detection
from dogtracker_pc.frames import discover_frames
from dogtracker_pc.server import create_app


@pytest.fixture
def client(frames_folder: Path):
    frames = discover_frames(frames_folder)
    dets = [
        Detection(frames[i].filename, frames[i].timestamp_ms, frames[i].width, frames[i].height, 10 + i, 10, 5, 5, 0.8)
        for i in range(3)
    ]
    app = create_app(frames_folder, frames, dets)
    app.testing = True
    return app.test_client(), frames, dets


def test_index_serves_dashboard(client):
    c, _, _ = client
    res = c.get("/")
    assert res.status_code == 200
    assert b"Dog Trajectory Tracker" in res.data


def test_static_assets_served(client):
    c, _, _ = client
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/static/style.css").status_code == 200


def test_summary_endpoint(client):
    c, frames, dets = client
    data = c.get("/api/summary").get_json()
    assert data["frame_count"] == len(frames)
    assert data["detection_count"] == len(dets)
    assert data["frame_width"] == frames[0].width
    assert data["duration_ms"] == frames[-1].timestamp_ms - frames[0].timestamp_ms
    assert data["reference_frame"] == frames[len(frames) // 2].filename


def test_areas_endpoint_ranks_and_includes_representative_frames(client):
    c, _, dets = client
    data = c.get("/api/areas?distance=1000&gap=100000&area_radius=1000").get_json()
    # With very loose thresholds every detection folds into one visit and area.
    assert len(data) == 1
    area = data[0]
    assert area["rank"] == 1
    assert area["visit_count"] == 1
    assert len(area["visits"]) == 1
    assert area["visits"][0]["representative_filename"] in {d.filename for d in dets}


def test_detections_endpoint(client):
    c, _, dets = client
    data = c.get("/api/detections").get_json()
    assert len(data) == len(dets)
    assert data[0]["filename"] == dets[0].filename


def test_visits_endpoint_respects_query_params(client):
    c, _, _ = client
    tight = c.get("/api/visits?distance=1&gap=100").get_json()
    loose = c.get("/api/visits?distance=1000&gap=100000").get_json()
    assert len(loose) <= len(tight)


def test_heatmap_png(client):
    c, _, _ = client
    res = c.get("/api/heatmap.png")
    assert res.status_code == 200
    assert res.content_type == "image/png"
    assert res.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_nearest_endpoint(client):
    c, _, dets = client
    res = c.get(f"/api/nearest?x={dets[1].x}&y={dets[1].y}")
    assert res.get_json()["index"] == 1


def test_nearest_requires_coordinates(client):
    c, _, _ = client
    assert c.get("/api/nearest").status_code == 400


def test_serve_known_frame(client):
    c, frames, _ = client
    res = c.get(f"/frames/{frames[0].filename}")
    assert res.status_code == 200
    assert res.content_type == "image/jpeg"


def test_serve_unknown_frame_404s(client):
    c, _, _ = client
    assert c.get("/frames/does-not-exist.jpg").status_code == 404


def test_path_traversal_is_blocked(client):
    c, _, _ = client
    res = c.get("/frames/..%2F..%2F..%2Fetc%2Fpasswd")
    assert res.status_code == 404


def test_empty_session_has_zero_counts(tmp_path: Path):
    frames = discover_frames(tmp_path)
    app = create_app(tmp_path, frames, [])
    c = app.test_client()
    data = c.get("/api/summary").get_json()
    assert data["frame_count"] == 0
    assert data["detection_count"] == 0
    assert c.get("/api/heatmap.png").status_code == 200
