from pathlib import Path

import numpy as np

from dogtracker_pc.detect import Detection, load_cache, run_detection
from dogtracker_pc.frames import discover_frames


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
    """Mimics ultralytics.YOLO enough for detect._detect_single to work.

    ``boxes_by_filename`` maps a frame filename to a list of (xyxy, conf)
    box tuples; a filename absent from the map (or mapped to []) yields no
    detection, matching a frame with no dog in it.
    """

    def __init__(self, boxes_by_filename: dict):
        self.boxes_by_filename = boxes_by_filename
        self.predict_calls = 0

    def predict(self, source, classes, verbose):
        self.predict_calls += 1
        filename = Path(source).name
        entries = self.boxes_by_filename.get(filename, [])
        if not entries:
            return [_FakeResult(_FakeBoxes([], []))]
        xyxy = [e[0] for e in entries]
        conf = [e[1] for e in entries]
        return [_FakeResult(_FakeBoxes(xyxy, conf))]


def test_run_detection_picks_highest_confidence_box(frames_folder: Path):
    frames = discover_frames(frames_folder)
    first = frames[0].filename
    model = _FakeModel(
        {
            first: [
                ([10, 10, 20, 20], 0.4),
                ([30, 20, 50, 60], 0.9),  # this one should win
            ]
        }
    )
    detections = run_detection(frames_folder, frames, model=model, use_cache=False)
    assert len(detections) == 1
    det = detections[0]
    assert det.filename == first
    assert det.confidence == 0.9
    assert det.x == 40.0  # (30+50)/2
    assert det.y == 40.0  # (20+60)/2
    assert det.w == 20.0
    assert det.h == 40.0


def test_run_detection_skips_frames_with_no_dog(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({})
    detections = run_detection(frames_folder, frames, model=model, use_cache=False)
    assert detections == []


def test_cache_avoids_rerunning_model(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({f.filename: [([0, 0, 10, 10], 0.5)] for f in frames})

    first_pass = run_detection(frames_folder, frames, model=model, use_cache=True)
    assert len(first_pass) == len(frames)
    assert model.predict_calls == len(frames)

    cache = load_cache(frames_folder)
    assert set(cache.keys()) == {f.filename for f in frames}

    # Second run should hit the cache entirely -- model.predict must not be
    # called again, and passing model=None must not force a (real) model load.
    second_pass = run_detection(frames_folder, frames, model=None, use_cache=True)
    assert len(second_pass) == len(frames)
    assert model.predict_calls == len(frames)  # unchanged


def test_cache_invalidated_when_file_changes(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({f.filename: [] for f in frames})
    run_detection(frames_folder, frames, model=model, use_cache=True)
    assert model.predict_calls == len(frames)

    # Touch one frame's content (changes size -> fingerprint changes).
    changed = frames[0].path
    changed.write_bytes(changed.read_bytes() + b"\x00")

    frames_again = discover_frames(frames_folder)
    model2 = _FakeModel({f.filename: [([1, 1, 5, 5], 0.6)] for f in frames_again})
    detections = run_detection(frames_folder, frames_again, model=model2, use_cache=True)

    assert model2.predict_calls == 1  # only the changed frame was re-run
    assert len(detections) == 1
    assert detections[0].filename == changed.name


def test_detection_is_a_plain_dataclass():
    det = Detection("a.jpg", 0, 64, 48, 1.0, 2.0, 3.0, 4.0, 0.9)
    assert det.filename == "a.jpg"
