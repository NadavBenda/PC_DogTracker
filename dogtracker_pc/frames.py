"""Discovery of JPEG frames recorded by the ESP32 to its SD card."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

# The firmware names snapshots like "dog_<millis-or-epoch>.jpg". Grab the
# digit run in the filename as the timestamp; fall back to file mtime for
# frames that were renamed or came from elsewhere. No minimum digit count:
# millis() is only 5 digits (or fewer) for the first ~100 seconds after
# boot, and requiring 6+ silently mis-sorted exactly those early frames by
# file mtime instead (which reflects copy order, not capture order).
_TIMESTAMP_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class Frame:
    """A single frame image discovered on disk."""

    path: Path
    filename: str
    timestamp_ms: int
    width: int
    height: int
    mtime: float
    size: int


def _parse_timestamp_ms(path: Path) -> int:
    match = _TIMESTAMP_RE.search(path.stem)
    if match:
        return int(match.group(1))
    return int(path.stat().st_mtime * 1000)


def discover_frames(folder: Path) -> list[Frame]:
    """Scan ``folder`` (non-recursive) for JPEG frames, sorted by timestamp."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    frames: list[Frame] = []
    for entry in sorted(folder.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(entry) as img:
                width, height = img.size
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning("Skipping unreadable image %s: %s", entry.name, exc)
            continue

        stat = entry.stat()
        frames.append(
            Frame(
                path=entry,
                filename=entry.name,
                timestamp_ms=_parse_timestamp_ms(entry),
                width=width,
                height=height,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        )

    frames.sort(key=lambda f: (f.timestamp_ms, f.filename))
    return frames
