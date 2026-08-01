import os
import time
from pathlib import Path

import pytest
from PIL import Image

from dogtracker_pc.frames import discover_frames


def test_discovers_and_sorts_by_timestamp(frames_folder: Path):
    frames = discover_frames(frames_folder)
    assert len(frames) == 6
    assert [f.timestamp_ms for f in frames] == sorted(f.timestamp_ms for f in frames)
    assert frames[0].timestamp_ms == 1_000_000
    assert frames[0].width == 64
    assert frames[0].height == 48


def test_ignores_non_image_files(frames_folder: Path):
    (frames_folder / "notes.txt").write_text("hello")
    (frames_folder / "detections.json").write_text("{}")
    frames = discover_frames(frames_folder)
    assert len(frames) == 6


def test_skips_unreadable_images(frames_folder: Path):
    corrupt = frames_folder / "dog_9999999.jpg"
    corrupt.write_bytes(b"not a real jpeg")
    frames = discover_frames(frames_folder)
    assert len(frames) == 6
    assert "dog_9999999.jpg" not in {f.filename for f in frames}


def test_falls_back_to_mtime_when_no_digits_in_name(tmp_path: Path):
    img = Image.new("RGB", (32, 32))
    path = tmp_path / "frame.jpg"
    img.save(path)
    frames = discover_frames(tmp_path)
    assert len(frames) == 1
    assert frames[0].timestamp_ms > 0


def test_short_millis_timestamps_sort_correctly_not_by_mtime(tmp_path: Path):
    # dog_99725.jpg (5 digits, <100s after boot) must sort before
    # dog_100225.jpg (6 digits) by its real timestamp -- even with mtimes
    # backwards, which is exactly what happens when files are copied off
    # the SD card in a different order than they were captured.
    early = tmp_path / "dog_99725.jpg"
    later = tmp_path / "dog_100225.jpg"
    Image.new("RGB", (16, 16)).save(later)
    Image.new("RGB", (16, 16)).save(early)
    now = time.time()
    os.utime(later, (now - 10, now - 10))
    os.utime(early, (now, now))

    frames = discover_frames(tmp_path)

    assert [f.filename for f in frames] == ["dog_99725.jpg", "dog_100225.jpg"]
    assert frames[0].timestamp_ms == 99725
    assert frames[1].timestamp_ms == 100225


def test_raises_for_missing_folder(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        discover_frames(tmp_path / "does_not_exist")
