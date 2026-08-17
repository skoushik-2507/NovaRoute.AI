"""
tests/test_tracking.py

Unit tests for ml/src/tracking/vehicle_tracker.py.

These do NOT construct a real VehicleDetector (which would load actual
YOLO weights) and do NOT need ByteTrack/ultralytics to run inference.
Instead, a lightweight fake stands in for both the detector and its
underlying `model.track()` call, mirroring the fake-object pattern already
used in tests/test_counting.py. This keeps the tests fast, offline, and
focused purely on VehicleTracker's own logic: turning Ultralytics track
results into TrackedVehicle objects.

Run with: pytest tests/test_tracking.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from src.tracking.vehicle_tracker import VehicleTracker, TrackedVehicle  # noqa: E402


# --- fakes -------------------------------------------------------------
# Minimal stand-ins for the Ultralytics objects VehicleTracker.track()
# reads from. Each only implements what track() actually touches.


class FakeXYXY:
    """Stands in for a single box's `.xyxy[0]` tensor (needs .tolist())."""

    def __init__(self, coords):
        self._coords = coords

    def tolist(self):
        return list(self._coords)


class FakeBox:
    """Stands in for one Ultralytics `Boxes` row."""

    def __init__(self, cls_id, conf, xyxy, track_id):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [FakeXYXY(xyxy)]
        self.id = [track_id] if track_id is not None else None


class FakeBoxes(list):
    """A plain list already satisfies `for box in result.boxes` and
    `result.boxes is None` checks (as long as we pass None explicitly when
    we want to simulate "no boxes")."""


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Stands in for `VehicleDetector.model`. Records the kwargs it was
    called with so tests can assert VehicleTracker wires config through
    correctly, and returns a preset list of FakeResult."""

    def __init__(self, results):
        self._results = results
        self.last_call_kwargs = None
        self.predictor = None

    def track(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._results


class FakeDetector:
    """Stands in for VehicleDetector: only the attributes VehicleTracker
    actually reads (model, target_classes, thresholds, device,
    _id_to_name). Deliberately does NOT import or construct a real
    VehicleDetector, so no YOLO weights are loaded."""

    def __init__(self, results, target_classes=None):
        self.target_classes = target_classes or {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
        self._id_to_name = {v: k for k, v in self.target_classes.items()}
        self.confidence_threshold = 0.35
        self.iou_threshold = 0.45
        self.device = "cpu"
        self.model = FakeModel(results)


# --- TrackedVehicle dataclass ------------------------------------------


def test_tracked_vehicle_dataclass_fields():
    tv = TrackedVehicle(
        track_id=7,
        class_name="car",
        confidence=0.91,
        bounding_box=(1.0, 2.0, 3.0, 4.0),
        frame_number=12,
    )
    assert tv.track_id == 7
    assert tv.class_name == "car"
    assert tv.confidence == 0.91
    assert tv.bounding_box == (1.0, 2.0, 3.0, 4.0)
    assert tv.frame_number == 12


def test_tracked_vehicle_to_dict():
    tv = TrackedVehicle(track_id=3, class_name="bus", confidence=0.8, bounding_box=(0, 0, 1, 1), frame_number=1)
    assert tv.to_dict() == {
        "track_id": 3,
        "class_name": "bus",
        "confidence": 0.8,
        "bounding_box": (0, 0, 1, 1),
        "frame_number": 1,
    }


# --- VehicleTracker.track() ---------------------------------------------


def test_track_returns_empty_when_no_results():
    detector = FakeDetector(results=[])
    tracker = VehicleTracker(detector)
    assert tracker.track(frame=None, frame_number=1) == []


def test_track_returns_empty_when_boxes_none():
    detector = FakeDetector(results=[FakeResult(boxes=None)])
    tracker = VehicleTracker(detector)
    assert tracker.track(frame=None, frame_number=1) == []


def test_track_parses_boxes_with_track_ids():
    boxes = FakeBoxes(
        [
            FakeBox(cls_id=2, conf=0.9, xyxy=(10, 20, 30, 40), track_id=101),
            FakeBox(cls_id=7, conf=0.8, xyxy=(50, 60, 70, 80), track_id=102),
        ]
    )
    detector = FakeDetector(results=[FakeResult(boxes=boxes)])
    tracker = VehicleTracker(detector)

    tracked = tracker.track(frame=None, frame_number=5)

    assert len(tracked) == 2
    assert tracked[0] == TrackedVehicle(
        track_id=101, class_name="car", confidence=0.9, bounding_box=(10.0, 20.0, 30.0, 40.0), frame_number=5
    )
    assert tracked[1] == TrackedVehicle(
        track_id=102, class_name="truck", confidence=0.8, bounding_box=(50.0, 60.0, 70.0, 80.0), frame_number=5
    )


def test_track_id_is_none_when_bytetrack_has_not_assigned_one():
    boxes = FakeBoxes([FakeBox(cls_id=2, conf=0.9, xyxy=(0, 0, 1, 1), track_id=None)])
    detector = FakeDetector(results=[FakeResult(boxes=boxes)])
    tracker = VehicleTracker(detector)

    tracked = tracker.track(frame=None, frame_number=1)

    assert len(tracked) == 1
    assert tracked[0].track_id is None
    # Vehicle is still reported (not dropped) even without a stable ID yet.
    assert tracked[0].class_name == "car"


def test_track_ignores_unmapped_class_ids():
    # cls_id 99 is not in target_classes -> should be silently skipped,
    # matching VehicleDetector's own "shouldn't happen but guard anyway"
    # behaviour.
    boxes = FakeBoxes(
        [
            FakeBox(cls_id=99, conf=0.5, xyxy=(0, 0, 1, 1), track_id=1),
            FakeBox(cls_id=2, conf=0.5, xyxy=(0, 0, 1, 1), track_id=2),
        ]
    )
    detector = FakeDetector(results=[FakeResult(boxes=boxes)])
    tracker = VehicleTracker(detector)

    tracked = tracker.track(frame=None, frame_number=1)

    assert len(tracked) == 1
    assert tracked[0].class_name == "car"


def test_track_passes_config_through_to_model_track():
    detector = FakeDetector(results=[FakeResult(boxes=FakeBoxes([]))])
    tracker = VehicleTracker(detector, tracker_config="bytetrack.yaml")

    tracker.track(frame="FRAME", frame_number=1)

    kwargs = detector.model.last_call_kwargs
    assert kwargs["tracker"] == "bytetrack.yaml"
    assert kwargs["persist"] is True
    assert kwargs["conf"] == detector.confidence_threshold
    assert kwargs["iou"] == detector.iou_threshold
    assert kwargs["device"] == detector.device
    assert set(kwargs["classes"]) == set(detector.target_classes.values())
    assert kwargs["source"] == "FRAME"


def test_tracker_overrides_take_precedence_over_detector_defaults():
    detector = FakeDetector(results=[FakeResult(boxes=FakeBoxes([]))])
    tracker = VehicleTracker(detector, confidence_threshold=0.9, iou_threshold=0.1, device="cuda:0")

    tracker.track(frame=None, frame_number=1)

    kwargs = detector.model.last_call_kwargs
    assert kwargs["conf"] == 0.9
    assert kwargs["iou"] == 0.1
    assert kwargs["device"] == "cuda:0"


# --- VehicleTracker.unique_track_ids() -----------------------------------


def test_unique_track_ids_excludes_none_and_deduplicates():
    vehicles = [
        TrackedVehicle(track_id=1, class_name="car", confidence=0.9, bounding_box=(0, 0, 1, 1), frame_number=1),
        TrackedVehicle(track_id=1, class_name="car", confidence=0.9, bounding_box=(0, 0, 1, 1), frame_number=2),
        TrackedVehicle(track_id=2, class_name="bus", confidence=0.9, bounding_box=(0, 0, 1, 1), frame_number=2),
        TrackedVehicle(track_id=None, class_name="truck", confidence=0.5, bounding_box=(0, 0, 1, 1), frame_number=2),
    ]
    assert VehicleTracker.unique_track_ids(vehicles) == [1, 2]


def test_unique_track_ids_empty_list():
    assert VehicleTracker.unique_track_ids([]) == []


# --- VehicleTracker.reset() ----------------------------------------------


class FakeSubTracker:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True


def test_reset_resets_underlying_bytetrack_state():
    detector = FakeDetector(results=[])
    tracker = VehicleTracker(detector)

    fake_predictor = type("FakePredictor", (), {})()
    fake_sub_tracker = FakeSubTracker()
    fake_predictor.trackers = [fake_sub_tracker]
    detector.model.predictor = fake_predictor

    tracker.reset()

    assert fake_sub_tracker.reset_called is True


def test_reset_is_a_noop_when_no_predictor_yet():
    detector = FakeDetector(results=[])
    tracker = VehicleTracker(detector)
    # model.predictor is None (never ran inference yet) - should not raise.
    tracker.reset()