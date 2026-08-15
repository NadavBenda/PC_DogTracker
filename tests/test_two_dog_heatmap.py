import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dogtracker_pc.detect import MIN_DETECTION_CONFIDENCE, Detection
from dogtracker_pc.frames import discover_frames

# two_dog_heatmap.py is a root-level standalone script (like debug_detect.py),
# not part of the dogtracker_pc package, so it's loaded directly by file path
# rather than via a normal package import.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "two_dog_heatmap.py"
_spec = importlib.util.spec_from_file_location("two_dog_heatmap", _SCRIPT_PATH)
two_dog_heatmap = importlib.util.module_from_spec(_spec)
sys.modules["two_dog_heatmap"] = two_dog_heatmap
_spec.loader.exec_module(two_dog_heatmap)


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = np.array(xyxy, dtype=np.float64)
        self.conf = np.array(conf, dtype=np.float64)

    def __len__(self):
        return len(self.conf)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """boxes_by_filename maps a filename to a list of (xyxy, conf) tuples --
    unlike detect.py's tests, here we care that MULTIPLE boxes per frame all
    survive, not just the highest-confidence one."""

    def __init__(self, boxes_by_filename: dict):
        self.boxes_by_filename = boxes_by_filename
        self.predict_calls = 0
        self.received_confs = []

    def predict(self, source, classes, verbose, conf=None):
        self.predict_calls += 1
        self.received_confs.append(conf)
        filename = Path(source).name if isinstance(source, (str, Path)) else None
        entries = self.boxes_by_filename.get(filename, []) if filename else []
        if not entries:
            return [_FakeResult(_FakeBoxes([], []))]
        xyxy = [e[0] for e in entries]
        confs = [e[1] for e in entries]
        return [_FakeResult(_FakeBoxes(xyxy, confs))]


def test_detect_all_dogs_keeps_every_box_above_threshold(frames_folder: Path):
    frames = discover_frames(frames_folder)
    first = frames[0]
    model = _FakeModel(
        {
            first.filename: [
                ([0, 0, 10, 10], 0.9),
                ([20, 20, 40, 50], 0.5),
            ]
        }
    )
    dets = two_dog_heatmap.detect_all_dogs(model, first, rotate_degrees=0)
    assert len(dets) == 2
    confidences = sorted(d.confidence for d in dets)
    assert confidences == [0.5, 0.9]
    # Centers computed correctly for each box independently.
    xs = sorted(d.x for d in dets)
    assert xs == [5.0, 30.0]


def test_detect_all_dogs_no_dog_returns_empty_list(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({})
    assert two_dog_heatmap.detect_all_dogs(model, frames[0]) == []


def test_detect_all_dogs_passes_minimum_confidence_to_model(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({})
    two_dog_heatmap.detect_all_dogs(model, frames[0])
    assert model.received_confs[-1] == MIN_DETECTION_CONFIDENCE


def test_run_multi_dog_detection_accumulates_across_frames(frames_folder: Path):
    frames = discover_frames(frames_folder)
    boxes = {
        frames[0].filename: [([0, 0, 10, 10], 0.9)],  # 1 dog
        frames[1].filename: [([0, 0, 10, 10], 0.9), ([50, 50, 70, 70], 0.6)],  # 2 dogs
        # frames[2..]: no dogs configured -> none detected
    }
    model = _FakeModel(boxes)
    detections = two_dog_heatmap.run_multi_dog_detection(frames_folder, frames, model, use_cache=False)
    assert len(detections) == 3
    assert two_dog_heatmap.frames_with_multiple_dogs(detections) == 1


def test_run_multi_dog_detection_caching_avoids_rerun(frames_folder: Path):
    frames = discover_frames(frames_folder)
    boxes = {f.filename: [([0, 0, 10, 10], 0.9), ([20, 20, 30, 30], 0.7)] for f in frames}
    model = _FakeModel(boxes)

    first_pass = two_dog_heatmap.run_multi_dog_detection(frames_folder, frames, model, use_cache=True)
    assert len(first_pass) == 2 * len(frames)
    assert model.predict_calls == len(frames)

    cache = two_dog_heatmap._load_cache(frames_folder, rotate_degrees=0)
    assert set(cache.keys()) == {f.filename for f in frames}

    second_pass = two_dog_heatmap.run_multi_dog_detection(frames_folder, frames, model, use_cache=True)
    assert len(second_pass) == 2 * len(frames)
    assert model.predict_calls == len(frames)  # unchanged -- cache hit


def test_multi_dog_cache_is_separate_from_main_detection_cache(frames_folder: Path):
    """The whole point of this script is to not disturb the main dashboard's
    pipeline -- its cache file must be untouched."""
    main_cache_dir = frames_folder / ".dogtracker_cache"
    main_cache_dir.mkdir()
    main_cache_path = main_cache_dir / "detections.json"
    main_cache_path.write_text('{"version": 1, "entries": {"sentinel": true}}')

    frames = discover_frames(frames_folder)
    model = _FakeModel({f.filename: [([0, 0, 10, 10], 0.9)] for f in frames})
    two_dog_heatmap.run_multi_dog_detection(frames_folder, frames, model, use_cache=True)

    # The main pipeline's cache file must be byte-for-byte untouched.
    assert main_cache_path.read_text() == '{"version": 1, "entries": {"sentinel": true}}'
    # This script's own cache lives in a different file.
    assert (main_cache_dir / "multi_dog_detections.json").exists()


def test_cache_invalidated_when_rotation_setting_changes(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({f.filename: [([0, 0, 10, 10], 0.5)] for f in frames})
    two_dog_heatmap.run_multi_dog_detection(frames_folder, frames, model, use_cache=True, rotate_degrees=0)
    assert model.predict_calls == len(frames)

    model_rotated = _FakeModel({f.filename: [([0, 0, 10, 10], 0.5)] for f in frames})
    two_dog_heatmap.run_multi_dog_detection(frames_folder, frames, model_rotated, use_cache=True, rotate_degrees=90)
    assert model_rotated.predict_calls == len(frames)


def test_cache_load_ignores_corrupt_file(frames_folder: Path):
    cache_dir = frames_folder / ".dogtracker_cache"
    cache_dir.mkdir()
    (cache_dir / "multi_dog_detections.json").write_text("not json")
    assert two_dog_heatmap._load_cache(frames_folder, rotate_degrees=0) == {}


def test_frames_with_multiple_dogs_counts_correctly():
    dets = [
        Detection("a.jpg", 0, 100, 100, 1, 1, 2, 2, 0.9),
        Detection("a.jpg", 0, 100, 100, 5, 5, 2, 2, 0.8),  # a.jpg has 2
        Detection("b.jpg", 100, 100, 100, 1, 1, 2, 2, 0.9),  # b.jpg has 1
    ]
    assert two_dog_heatmap.frames_with_multiple_dogs(dets) == 1


def test_render_heatmap_image_matches_reference_frame_size(frames_folder: Path):
    frames = discover_frames(frames_folder)  # 64x48 per the conftest fixture
    dets = [Detection(frames[0].filename, frames[0].timestamp_ms, 64, 48, 32, 24, 5, 5, 0.9)]
    image = two_dog_heatmap.render_heatmap_image(frames, dets, rotate_degrees=0)
    assert image.size == (64, 48)
    assert image.mode == "RGB"  # flattened/opaque, not RGBA


def test_render_heatmap_image_swaps_dimensions_for_90_degree_rotation(frames_folder: Path):
    frames = discover_frames(frames_folder)  # 64x48
    dets = [Detection(frames[0].filename, frames[0].timestamp_ms, 48, 64, 20, 30, 5, 5, 0.9)]
    image = two_dog_heatmap.render_heatmap_image(frames, dets, rotate_degrees=90)
    assert image.size == (48, 64)


def test_render_heatmap_image_raises_on_no_frames():
    with pytest.raises(ValueError):
        two_dog_heatmap.render_heatmap_image([], [])
