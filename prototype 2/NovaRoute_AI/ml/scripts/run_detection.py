"""
ml/scripts/run_detection.py

Command-line entry point for Phase 1 (YOLO vehicle detection) ONLY.

This file contains no business logic itself - it just:
  1. loads config.yaml
  2. constructs a VehicleDetector from it
  3. reads frames from the given source
  4. calls detector.detect() per frame
  5. draws + writes the annotated output
  6. prints a summary

All detection logic lives in ml/src/detection/vehicle_detector.py.
No tracking, congestion, risk scoring, FastAPI, or frontend code here.

Usage:
    python ml/scripts/run_detection.py --source ml/data/test_videos/traffic_video.mp4
    python ml/scripts/run_detection.py --source path/to/image.jpg
    python ml/scripts/run_detection.py --source 0        # webcam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.detection.vehicle_detector import VehicleDetector  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# --- drawing helpers (kept local to this script since Phase 1 only needs
# them here; move to ml/src/utils/visualization.py if reused elsewhere later) ---

CLASS_COLORS = {
    "car": (60, 180, 75),
    "bus": (0, 130, 200),
    "truck": (245, 130, 48),
    "motorcycle": (145, 30, 180),
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bounding_box]
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def put_summary_text(frame, lines, origin=(10, 25)):
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# --- CLI plumbing ---


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_detector(cfg: dict) -> VehicleDetector:
    det_cfg = cfg["detection"]
    configured_model = str(det_cfg["model_weights"])
    model_path = Path(configured_model)
    if model_path.is_absolute() or model_path.parent != Path("."):
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        configured_model = str(model_path)
    return VehicleDetector(
        model_weights=configured_model,
        target_classes=det_cfg["target_classes"],
        confidence_threshold=det_cfg["confidence_threshold"],
        iou_threshold=det_cfg["iou_threshold"],
        device=det_cfg["device"],
    )


def run_on_image(detector: VehicleDetector, source_path: Path, out_path: Path, show: bool) -> None:
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {source_path}")

    detections = detector.detect(frame)
    counts = detector.count_by_class(detections)

    frame = draw_detections(frame, detections)
    summary_lines = [f"{cls}: {n}" for cls, n in counts.items() if cls != "total"]
    summary_lines.append(f"Total: {counts.get('total', 0)}")
    frame = put_summary_text(frame, summary_lines)

    cv2.imwrite(str(out_path), frame)

    if show:
        cv2.imshow("NovaRoute.AI - Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n=== Detection summary (single image) ===")
    for cls_name in ["car", "bus", "truck", "motorcycle"]:
        print(f"{cls_name.capitalize()}s: {counts.get(cls_name, 0)}")
    print(f"Total: {counts.get('total', 0)}")
    print(f"\nAnnotated image written to: {out_path}")


def run_on_video(detector: VehicleDetector, source, out_path: Path, show: bool) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {out_path}")

    sum_counts: dict[str, int] = {}
    peak_total = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        detections = detector.detect(frame)
        counts = detector.count_by_class(detections)
        for cls_name, n in counts.items():
            if cls_name == "total":
                continue
            sum_counts[cls_name] = sum_counts.get(cls_name, 0) + n
        peak_total = max(peak_total, counts.get("total", 0))

        frame = draw_detections(frame, detections)
        summary_lines = [f"Frame {frame_idx}"]
        summary_lines += [f"{cls}: {n}" for cls, n in counts.items() if cls != "total"]
        summary_lines.append(f"Total: {counts.get('total', 0)}")
        frame = put_summary_text(frame, summary_lines)

        writer.write(frame)
        if show:
            cv2.imshow("NovaRoute.AI - Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n=== Detection summary (frame observations) ===")
    if frame_idx:
        for cls_name in ["car", "bus", "truck", "motorcycle"]:
            avg = sum_counts.get(cls_name, 0) / frame_idx
            print(f"Average {cls_name}: {avg:.2f} per frame")
        print(f"Peak vehicles in a frame: {peak_total}")
    else:
        print("No frames were processed")
    print(f"\nAnnotated video written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: YOLO vehicle detection (image or video)")
    parser.add_argument("--source", required=True, help="Video/image file path, or webcam index")
    parser.add_argument("--config", default=str(ROOT / "ml/config/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--output", default=None, help="Output file path (defaults into ml/outputs/detections/)")
    parser.add_argument("--no-display", action="store_true", help="Don't open a preview window")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    detector = build_detector(cfg)

    out_dir = ROOT / cfg["io"]["detections_out"]
    out_dir.mkdir(parents=True, exist_ok=True)

    is_webcam = args.source.isdigit()
    source_path = None if is_webcam else Path(args.source)

    if source_path is not None and source_path.suffix.lower() in IMAGE_EXTENSIONS:
        out_path = Path(args.output) if args.output else out_dir / f"{source_path.stem}_detected.jpg"
        run_on_image(detector, source_path, out_path, show=not args.no_display)
    else:
        source = int(args.source) if is_webcam else args.source
        stem = "webcam" if is_webcam else source_path.stem
        out_path = Path(args.output) if args.output else out_dir / f"{stem}_detected.mp4"
        run_on_video(detector, source, out_path, show=not args.no_display)


if __name__ == "__main__":
    main()