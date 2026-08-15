from pathlib import Path

import numpy as np
from PIL import Image

from dogtracker_pc.detect import (
    MIN_DETECTION_CONFIDENCE,
    Detection,
    _detect_single,
    detect_all_dogs,
    frames_with_multiple_dogs,
    load_cache,
    load_multi_dog_cache,
    run_detection,
    run_multi_dog_detection,
)
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
        self.received_confs = []

    def predict(self, source, classes, verbose, conf=None):
        self.predict_calls += 1
        self.received_confs.append(conf)
        if isinstance(source, (str, Path)):
            filename = Path(source).name
            entries = self.boxes_by_filename.get(filename, [])
        else:
            # A rotated frame is handed over as an already-loaded image, not
            # a path; tests that rotate don't need per-filename lookup here.
            entries = next(iter(self.boxes_by_filename.values()), [])
        if not entries:
            return [_FakeResult(_FakeBoxes([], []))]
        xyxy = [e[0] for e in entries]
        conf = [e[1] for e in entries]
        return [_FakeResult(_FakeBoxes(xyxy, conf))]


def test_run_detection_picks_highest_confidence_box(frames_folder: Path):
    frames = discover_frames(frames_folder)
    first = frames[0].filename
    second = frames[1].filename  # gives `first` an adjacent-frame confirmation
    model = _FakeModel(
        {
            first: [
                ([10, 10, 20, 20], 0.4),
                ([30, 20, 50, 60], 0.9),  # this one should win
            ],
            second: [([0, 0, 5, 5], 0.5)],
        }
    )
    detections = run_detection(frames_folder, frames, model=model, use_cache=False)
    det = next(d for d in detections if d.filename == first)
    assert det.confidence == 0.9
    assert det.x == 40.0  # (30+50)/2
    assert det.y == 40.0  # (20+60)/2
    assert det.w == 20.0
    assert det.h == 40.0


def test_detect_single_passes_minimum_confidence_to_model(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FixedBoxModel([1, 1, 5, 5], 0.9)
    _detect_single(model, frames[0], rotate_degrees=0)
    assert model.received_confs[-1] == MIN_DETECTION_CONFIDENCE


def test_run_detection_drops_isolated_single_frame_detections(frames_folder: Path):
    frames = discover_frames(frames_folder)
    # frames[0]: isolated (frames[1] has no dog) -> dropped.
    # frames[2], frames[3]: a consecutive pair -> both kept.
    # frames[5]: isolated at the end (frames[4] has no dog) -> dropped.
    boxes = {
        frames[0].filename: [([0, 0, 10, 10], 0.9)],
        frames[2].filename: [([0, 0, 10, 10], 0.9)],
        frames[3].filename: [([0, 0, 10, 10], 0.9)],
        frames[5].filename: [([0, 0, 10, 10], 0.9)],
    }
    model = _FakeModel(boxes)
    detections = run_detection(frames_folder, frames, model=model, use_cache=False)
    assert {d.filename for d in detections} == {frames[2].filename, frames[3].filename}


def test_run_detection_consecutive_confirmation_ignores_position(frames_folder: Path):
    frames = discover_frames(frames_folder)
    # Two adjacent frames both have a dog, but at very different positions --
    # still confirmed, since confirmation is presence-only, not proximity-based.
    boxes = {
        frames[0].filename: [([0, 0, 5, 5], 0.9)],
        frames[1].filename: [([200, 150, 210, 160], 0.9)],
    }
    model = _FakeModel(boxes)
    detections = run_detection(frames_folder, frames, model=model, use_cache=False)
    assert {d.filename for d in detections} == {frames[0].filename, frames[1].filename}


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

    # Touch two adjacent frames' content (changes size -> fingerprint
    # changes) so both need re-running -- both are given a detection so
    # they confirm each other under the consecutive-frame requirement
    # (a single re-run frame with no confirming neighbor would otherwise be
    # dropped, muddying what this test is actually checking: that only the
    # changed files get re-detected).
    changed_a = frames[0].path
    changed_a.write_bytes(changed_a.read_bytes() + b"\x00")
    changed_b = frames[1].path
    changed_b.write_bytes(changed_b.read_bytes() + b"\x00")

    frames_again = discover_frames(frames_folder)
    model2 = _FakeModel(
        {
            frames_again[0].filename: [([1, 1, 5, 5], 0.6)],
            frames_again[1].filename: [([1, 1, 5, 5], 0.6)],
        }
    )
    detections = run_detection(frames_folder, frames_again, model=model2, use_cache=True)

    assert model2.predict_calls == 2  # only the two changed frames were re-run
    assert {d.filename for d in detections} == {changed_a.name, changed_b.name}


def test_detection_is_a_plain_dataclass():
    det = Detection("a.jpg", 0, 64, 48, 1.0, 2.0, 3.0, 4.0, 0.9)
    assert det.filename == "a.jpg"


class _FixedBoxModel:
    """Always returns the same single box, whatever `source` is -- used to
    check what _detect_single actually hands the model when rotating."""

    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes([xyxy], [conf])
        self.received_sources = []
        self.received_confs = []

    def predict(self, source, classes, verbose, conf=None):
        self.received_sources.append(source)
        self.received_confs.append(conf)
        return [_FakeResult(self.boxes)]


def test_detect_single_without_rotation_passes_the_file_path(frames_folder: Path):
    frames = discover_frames(frames_folder)
    frame = frames[0]  # 64x48 per the conftest fixture
    model = _FixedBoxModel([1, 1, 5, 5], 0.9)

    det = _detect_single(model, frame, rotate_degrees=0)

    assert isinstance(model.received_sources[-1], str)
    assert det.frame_width == 64
    assert det.frame_height == 48


def test_detect_single_with_90_degree_rotation_swaps_dimensions(frames_folder: Path):
    frames = discover_frames(frames_folder)
    frame = frames[0]  # 64x48
    model = _FixedBoxModel([1, 1, 5, 5], 0.9)

    det = _detect_single(model, frame, rotate_degrees=90)

    # A rotated frame is handed over as an already-loaded image, not a path,
    # and its dimensions are swapped relative to the file's own 64x48.
    assert isinstance(model.received_sources[-1], Image.Image)
    assert det.frame_width == 48
    assert det.frame_height == 64


def test_detect_single_with_180_degree_rotation_keeps_dimensions(frames_folder: Path):
    frames = discover_frames(frames_folder)
    frame = frames[0]  # 64x48
    model = _FixedBoxModel([1, 1, 5, 5], 0.9)

    det = _detect_single(model, frame, rotate_degrees=180)

    assert det.frame_width == 64
    assert det.frame_height == 48


def test_cache_is_invalidated_when_rotation_setting_changes(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({f.filename: [([0, 0, 10, 10], 0.5)] for f in frames})

    run_detection(frames_folder, frames, model=model, use_cache=True, rotate_degrees=0)
    assert model.predict_calls == len(frames)

    # Same frames, same model, but a different rotation setting -- the old
    # cache (built at rotate_degrees=0) must not be reused, since its box
    # coordinates are in a different coordinate space.
    model_rotated = _FakeModel({f.filename: [([0, 0, 10, 10], 0.5)] for f in frames})
    run_detection(frames_folder, frames, model=model_rotated, use_cache=True, rotate_degrees=90)
    assert model_rotated.predict_calls == len(frames)


# ======================================================
# Multi-dog path (detect_all_dogs / run_multi_dog_detection): keeps every
# box per frame, unlike the single-dog path above which keeps only the best
# one. Separate cache file (multi_dog_detections.json), no consecutive-frame
# confirmation filter.
# ======================================================
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
    dets = detect_all_dogs(model, first, rotate_degrees=0)
    assert len(dets) == 2
    assert sorted(d.confidence for d in dets) == [0.5, 0.9]
    assert sorted(d.x for d in dets) == [5.0, 30.0]


def test_detect_all_dogs_no_dog_returns_empty_list(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({})
    assert detect_all_dogs(model, frames[0]) == []


def test_detect_all_dogs_passes_minimum_confidence_to_model(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({})
    detect_all_dogs(model, frames[0])
    assert model.received_confs[-1] == MIN_DETECTION_CONFIDENCE


def test_run_multi_dog_detection_accumulates_across_frames(frames_folder: Path):
    frames = discover_frames(frames_folder)
    boxes = {
        frames[0].filename: [([0, 0, 10, 10], 0.9)],  # 1 dog
        frames[1].filename: [([0, 0, 10, 10], 0.9), ([50, 50, 70, 70], 0.6)],  # 2 dogs
    }
    model = _FakeModel(boxes)
    detections = run_multi_dog_detection(frames_folder, frames, model=model, use_cache=False)
    assert len(detections) == 3
    assert frames_with_multiple_dogs(detections) == 1


def test_run_multi_dog_detection_no_consecutive_frame_filter(frames_folder: Path):
    """Unlike run_detection(), an isolated single-frame detection is kept --
    the confirmation heuristic assumes one continuous subject and doesn't
    apply when there may be more than one animal in view only briefly."""
    frames = discover_frames(frames_folder)
    boxes = {frames[0].filename: [([0, 0, 10, 10], 0.9)]}  # only frame with a dog, isolated
    model = _FakeModel(boxes)
    detections = run_multi_dog_detection(frames_folder, frames, model=model, use_cache=False)
    assert len(detections) == 1


def test_run_multi_dog_detection_caching_avoids_rerun(frames_folder: Path):
    frames = discover_frames(frames_folder)
    boxes = {f.filename: [([0, 0, 10, 10], 0.9), ([20, 20, 30, 30], 0.7)] for f in frames}
    model = _FakeModel(boxes)

    first_pass = run_multi_dog_detection(frames_folder, frames, model=model, use_cache=True)
    assert len(first_pass) == 2 * len(frames)
    assert model.predict_calls == len(frames)

    cache = load_multi_dog_cache(frames_folder)
    assert set(cache.keys()) == {f.filename for f in frames}

    second_pass = run_multi_dog_detection(frames_folder, frames, model=model, use_cache=True)
    assert len(second_pass) == 2 * len(frames)
    assert model.predict_calls == len(frames)  # unchanged -- cache hit


def test_multi_dog_cache_is_separate_from_single_dog_cache(frames_folder: Path):
    frames = discover_frames(frames_folder)

    single_model = _FakeModel({f.filename: [([0, 0, 10, 10], 0.9)] for f in frames})
    run_detection(frames_folder, frames, model=single_model, use_cache=True)
    single_cache_path = frames_folder / ".dogtracker_cache" / "detections.json"
    before = single_cache_path.read_text()

    multi_model = _FakeModel({f.filename: [([0, 0, 10, 10], 0.9), ([20, 20, 30, 30], 0.5)] for f in frames})
    run_multi_dog_detection(frames_folder, frames, model=multi_model, use_cache=True)

    # The single-dog cache file must be byte-for-byte untouched by a
    # multi-dog run, and vice versa is exercised implicitly since each
    # writes to its own file.
    assert single_cache_path.read_text() == before
    assert (frames_folder / ".dogtracker_cache" / "multi_dog_detections.json").exists()


def test_multi_dog_cache_invalidated_when_rotation_setting_changes(frames_folder: Path):
    frames = discover_frames(frames_folder)
    model = _FakeModel({f.filename: [([0, 0, 10, 10], 0.5)] for f in frames})
    run_multi_dog_detection(frames_folder, frames, model=model, use_cache=True, rotate_degrees=0)
    assert model.predict_calls == len(frames)

    model_rotated = _FakeModel({f.filename: [([0, 0, 10, 10], 0.5)] for f in frames})
    run_multi_dog_detection(frames_folder, frames, model=model_rotated, use_cache=True, rotate_degrees=90)
    assert model_rotated.predict_calls == len(frames)


def test_multi_dog_cache_load_ignores_corrupt_file(frames_folder: Path):
    cache_dir = frames_folder / ".dogtracker_cache"
    cache_dir.mkdir()
    (cache_dir / "multi_dog_detections.json").write_text("not json")
    assert load_multi_dog_cache(frames_folder) == {}


def test_frames_with_multiple_dogs_counts_correctly():
    dets = [
        Detection("a.jpg", 0, 100, 100, 1, 1, 2, 2, 0.9),
        Detection("a.jpg", 0, 100, 100, 5, 5, 2, 2, 0.8),  # a.jpg has 2
        Detection("b.jpg", 100, 100, 100, 1, 1, 2, 2, 0.9),  # b.jpg has 1
    ]
    assert frames_with_multiple_dogs(dets) == 1
