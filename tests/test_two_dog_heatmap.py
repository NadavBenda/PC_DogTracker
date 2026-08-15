import importlib.util
import sys
from pathlib import Path

import pytest

import dogtracker_pc.detect as detect
from dogtracker_pc.detect import Detection
from dogtracker_pc.frames import discover_frames

# two_dog_heatmap.py is a root-level standalone script (like debug_detect.py),
# not part of the dogtracker_pc package, so it's loaded directly by file path
# rather than via a normal package import.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "two_dog_heatmap.py"
_spec = importlib.util.spec_from_file_location("two_dog_heatmap", _SCRIPT_PATH)
two_dog_heatmap = importlib.util.module_from_spec(_spec)
sys.modules["two_dog_heatmap"] = two_dog_heatmap
_spec.loader.exec_module(two_dog_heatmap)


def test_detection_functions_are_the_same_objects_as_in_detect_module():
    """The script re-uses dogtracker_pc.detect's multi-dog functions rather
    than keeping its own copy -- this guards against that drifting back into
    a duplicate implementation."""
    assert two_dog_heatmap.run_multi_dog_detection is detect.run_multi_dog_detection
    assert two_dog_heatmap.frames_with_multiple_dogs is detect.frames_with_multiple_dogs


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
