from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def frames_folder(tmp_path: Path) -> Path:
    """A folder of small synthetic JPEG frames, named like the firmware's snapshots."""
    base_ts = 1_000_000
    for i in range(6):
        img = Image.new("RGB", (64, 48), color=(20, 20, 20))
        img.save(tmp_path / f"dog_{base_ts + i * 500}.jpg", quality=80)
    return tmp_path
