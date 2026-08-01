"""Standalone YOLO diagnostic tool -- run directly on 1-3 real images to see
exactly what the model detects (at any confidence, in any class), without
the caching/threshold logic the full dogtracker_pc pipeline applies.

Useful when the full tool finds zero dogs in a whole session: this shows
whether the model is missing the dog completely, or "almost" seeing it at
a confidence just below the pipeline's implicit cutoff -- which points at
very different fixes (focus/distance/lighting vs. a threshold tweak).

Usage:
    python debug_detect.py photo1.jpg [photo2.jpg photo3.jpg ...]

For each image, prints every detected object (any class, down to a very
low confidence) and saves an annotated copy with boxes drawn next to the
original, e.g. photo1_annotated.jpg.
"""

import sys
from pathlib import Path

# Near-zero on purpose: the point is to see everything the model considered,
# not just what would normally be reported as a real detection.
MIN_CONFIDENCE = 0.001


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python debug_detect.py photo1.jpg [photo2.jpg photo3.jpg ...]")
        return 1

    from ultralytics import YOLO

    print("Loading yolov8s.pt (downloads once if not already cached locally)...")
    model = YOLO("yolov8s.pt")
    names = model.names

    for path_str in argv:
        path = Path(path_str)
        if not path.is_file():
            print(f"\n=== {path} === SKIPPED (not a file)")
            continue

        print(f"\n=== {path} ===")
        results = model.predict(source=str(path), conf=MIN_CONFIDENCE, verbose=False)
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print("  No objects detected at all, not even at near-zero confidence.")
            print("  That usually means the image is too blurry/dark/small for the")
            print("  model to find ANY recognizable shape -- not a dog-specific issue.")
        else:
            rows = sorted(
                zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()),
                key=lambda row: row[1],
                reverse=True,
            )
            for cls_id, conf, xyxy in rows:
                label = names[int(cls_id)]
                marker = "  <-- DOG" if label == "dog" else ""
                x1, y1, x2, y2 = (round(v) for v in xyxy)
                print(f"  {label:15s} conf={conf:.3f}  box=({x1},{y1})-({x2},{y2}){marker}")

        annotated_path = path.with_name(f"{path.stem}_annotated{path.suffix}")
        result.save(filename=str(annotated_path))
        print(f"  Annotated image saved to {annotated_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
