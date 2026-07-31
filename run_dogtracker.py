"""Launcher script used both for local `python run_dogtracker.py` and as the
PyInstaller entry point (see packaging/dogtracker.spec)."""

from dogtracker_pc.main import main

if __name__ == "__main__":
    raise SystemExit(main())
