"""One-click entry point: pick a folder, run detection, serve the dashboard."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Optional

from .detect import run_detection
from .frames import discover_frames
from .server import create_app

logger = logging.getLogger("dogtracker_pc")


def _find_free_port(preferred: int, attempts: int = 20) -> int:
    port = preferred
    for _ in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"Could not find a free port near {preferred}")


def _prompt_for_folder() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title="Select the folder of JPEG frames from the SD card")
        root.destroy()
    except Exception:
        logger.debug("Tk folder picker unavailable", exc_info=True)
        return None
    return Path(chosen) if chosen else None


class _ProgressWindow:
    """Best-effort Tk progress dialog shown while YOLO runs over new frames.

    Silently disables itself (falling back to console-only progress) on any
    Tk failure -- environments without a display are common (CI, SSH, this
    tool's own test suite) and must not crash the run.
    """

    def __init__(self, total: int):
        self._enabled = False
        try:
            import tkinter as tk
            from tkinter import ttk

            self.root = tk.Tk()
            self.root.title("Dog Trajectory Tracker")
            self.root.geometry("380x110")
            self.root.resizable(False, False)
            self.root.attributes("-topmost", True)

            self.label = tk.Label(self.root, text="Detecting dogs in frames...", anchor="w")
            self.label.pack(fill="x", padx=16, pady=(16, 8))

            self.bar = ttk.Progressbar(self.root, maximum=max(total, 1), length=340)
            self.bar.pack(padx=16)

            self._enabled = True
        except Exception:
            logger.debug("Tk progress window unavailable", exc_info=True)

    def update(self, done: int, total: int) -> None:
        if not self._enabled:
            return
        try:
            self.bar["maximum"] = max(total, 1)
            self.bar["value"] = done
            self.label.config(text=f"Detecting dogs in frames... {done}/{total}")
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self._enabled = False

    def close(self) -> None:
        if self._enabled:
            try:
                self.root.destroy()
            except Exception:
                pass


def _console_progress(done: int, total: int) -> None:
    pct = (done / total * 100) if total else 100.0
    sys.stdout.write(f"\rDetecting dogs in frames... {done}/{total} ({pct:0.1f}%)")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline dog trajectory analysis over ESP32 SD-card recordings.")
    parser.add_argument("folder", nargs="?", type=Path, help="Folder of JPEG frames to analyze")
    parser.add_argument("--port", type=int, default=5151, help="Preferred port for the local dashboard (default: 5151)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser window automatically")
    parser.add_argument(
        "--rescan", action="store_true", help="Ignore cached detections and re-run YOLO on every frame"
    )
    parser.add_argument("--no-gui", action="store_true", help="Never use Tk dialogs; FOLDER must be given")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    folder = args.folder
    if folder is None and not args.no_gui:
        folder = _prompt_for_folder()
    if folder is None:
        logger.error("No folder given and no folder was selected. Pass a folder path as an argument.")
        return 2

    folder = Path(folder)
    if not folder.is_dir():
        logger.error("Not a folder: %s", folder)
        return 2

    logger.info("Scanning %s for JPEG frames...", folder)
    frames = discover_frames(folder)
    logger.info("Found %d frames", len(frames))
    if not frames:
        logger.warning("No JPEG frames found -- the dashboard will still start, but will be empty.")

    progress_window = None if args.no_gui else _ProgressWindow(len(frames))

    def progress_cb(done: int, total: int) -> None:
        _console_progress(done, total)
        if progress_window:
            progress_window.update(done, total)

    logger.info("Running dog detection (cached results are reused automatically)...")
    detections = run_detection(folder, frames, progress_cb=progress_cb, use_cache=not args.rescan)
    logger.info("Detected the dog in %d/%d frames", len(detections), len(frames))

    if progress_window:
        progress_window.close()

    port = _find_free_port(args.port)
    app = create_app(folder, frames, detections)
    url = f"http://127.0.0.1:{port}/"

    if not args.no_browser:
        threading.Timer(0.75, lambda: webbrowser.open(url)).start()

    logger.info("Serving dashboard at %s (Ctrl+C to stop)", url)
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
