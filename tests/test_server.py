from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

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
    assert len(data["areas"]) == 1
    area = data["areas"][0]
    assert area["rank"] == 1
    assert area["visit_count"] == 1
    assert area["is_highlighted"] is True
    assert len(area["hull"]) >= 3
    assert len(area["visits"]) == 1
    assert area["visits"][0]["representative_filename"] in {d.filename for d in dets}
    assert data["total_visit_duration_ms"] == area["total_duration_ms"]
    assert data["elsewhere_duration_ms"] == 0


def test_areas_endpoint_top_n_marks_extra_areas_as_elsewhere(client):
    c, _, _ = client
    # Tight thresholds so the 3 detections (10,10)/(11,10)/(12,10) become 3
    # separate one-visit areas; top_n=1 highlights only the best one.
    data = c.get("/api/areas?distance=0&gap=100000&area_radius=0&top_n=1").get_json()
    assert len(data["areas"]) == 3
    assert data["areas"][0]["is_highlighted"] is True
    assert data["areas"][1]["is_highlighted"] is False
    assert data["areas"][2]["is_highlighted"] is False
    assert data["elsewhere_duration_ms"] == data["total_visit_duration_ms"] - data["areas"][0]["total_duration_ms"]


def test_detections_endpoint(client):
    c, _, dets = client
    data = c.get("/api/detections").get_json()
    assert len(data) == len(dets)
    assert data[0]["filename"] == dets[0].filename


def test_frames_endpoint_lists_every_frame_not_just_detections(client):
    c, frames, dets = client
    data = c.get("/api/frames").get_json()
    assert len(data) == len(frames)
    assert len(frames) > len(dets)  # fixture has more frames than detections
    assert [row["filename"] for row in data] == [f.filename for f in frames]


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


def test_rotated_summary_swaps_frame_dimensions(frames_folder: Path):
    frames = discover_frames(frames_folder)
    app = create_app(frames_folder, frames, [], rotate_degrees=90)
    c = app.test_client()
    data = c.get("/api/summary").get_json()
    assert data["frame_width"] == frames[0].height  # 48 -> width after a 90 deg turn
    assert data["frame_height"] == frames[0].width  # 64 -> height after a 90 deg turn


def test_rotated_frame_is_actually_rotated(frames_folder: Path):
    frames = discover_frames(frames_folder)
    app = create_app(frames_folder, frames, [], rotate_degrees=90)
    c = app.test_client()

    original_size = Image.open(frames[0].path).size  # (64, 48)
    res = c.get(f"/frames/{frames[0].filename}")
    assert res.status_code == 200
    assert res.content_type == "image/jpeg"

    served_size = Image.open(BytesIO(res.data)).size
    assert served_size == (original_size[1], original_size[0])  # (48, 64)


def test_unrotated_frame_uses_the_fast_send_from_directory_path(frames_folder: Path):
    frames = discover_frames(frames_folder)
    app = create_app(frames_folder, frames, [], rotate_degrees=0)
    c = app.test_client()
    res = c.get(f"/frames/{frames[0].filename}")
    assert res.status_code == 200
    assert Image.open(BytesIO(res.data)).size == (64, 48)


def test_empty_session_has_zero_counts(tmp_path: Path):
    frames = discover_frames(tmp_path)
    app = create_app(tmp_path, frames, [])
    c = app.test_client()
    data = c.get("/api/summary").get_json()
    assert data["frame_count"] == 0
    assert data["detection_count"] == 0
    assert c.get("/api/heatmap.png").status_code == 200
