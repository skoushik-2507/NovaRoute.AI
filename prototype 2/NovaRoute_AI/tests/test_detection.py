"""
tests/test_detection.py

Unit tests for ml/src/detection/vehicle_detector.py.

These only test the pure-logic pieces (class-id mapping, count
aggregation) and do NOT require YOLO weights to be downloaded, so they
run fast and offline. A VehicleDetector instance itself (which loads a
real model) is intentionally not constructed here.

Run with: pytest tests/test_detection.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from src.detection.vehicle_detector import (  # noqa: E402
    Detection,
    VehicleDetector,
    DEFAULT_TARGET_CLASSES,
)


def make_detection(class_name: str, class_id: int, conf: float = 0.9) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        bounding_box=(0.0, 0.0, 10.0, 10.0),
    )


def test_default_target_classes_match_coco_ids():
    assert DEFAULT_TARGET_CLASSES == {
        "car": 2,
        "motorcycle": 3,
        "bus": 5,
        "truck": 7,
    }


def test_detection_dataclass_fields():
    det = make_detection("car", 2, 0.87)
    assert det.class_id == 2
    assert det.class_name == "car"
    assert det.confidence == 0.87
    assert det.bounding_box == (0.0, 0.0, 10.0, 10.0)
    assert det.track_id is None  # not set at detection stage


def test_count_by_class_basic():
    detections = [
        make_detection("car", 2),
        make_detection("car", 2),
        make_detection("bus", 5),
        make_detection("truck", 7),
        make_detection("motorcycle", 3),
        make_detection("motorcycle", 3),
    ]
    counts = VehicleDetector.count_by_class(detections)
    assert counts["car"] == 2
    assert counts["bus"] == 1
    assert counts["truck"] == 1
    assert counts["motorcycle"] == 2
    assert counts["total"] == 6


def test_count_by_class_empty():
    counts = VehicleDetector.count_by_class([])
    assert counts == {"total": 0}