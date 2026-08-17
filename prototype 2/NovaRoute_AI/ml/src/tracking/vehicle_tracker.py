"""
ml/src/tracking/vehicle_tracker.py

Phase 2 - Vehicle tracking (ByteTrack).

Assigns a stable track_id to each vehicle across frames, on top of the
detections already produced by ml/src/detection/vehicle_detector.py.

This module does NOT re-implement YOLO inference or class-id filtering.
It wraps the same VehicleDetector used in Phase 1 (same model instance,
same target_classes / confidence / iou / device) and calls Ultralytics'
built-in `model.track()` API, which runs YOLO detection internally and
associates boxes across frames using ByteTrack. ByteTrack itself ships
inside the `ultralytics` package (as `bytetrack.yaml`) - this is the
"appropriate supported implementation" the task calls for, so no separate
ByteTrack library or custom re-identification code is written here.

No congestion, risk scoring, FastAPI, or frontend code in this file.
Downstream stages (ml/src/traffic/vehicle_counter.py today, a future
congestion module later) only ever see TrackedVehicle objects - they never
need to know ByteTrack was involved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.detection.vehicle_detector import VehicleDetector


@dataclass
class TrackedVehicle:
    """One tracked vehicle in one frame.

    This is the sole data contract downstream consumers (vehicle_counter.py,
    and later a congestion/density module) rely on - see the module
    docstring in vehicle_counter.py for how `class_name` alone is already
    consumed today, and how `track_id` + `frame_number` let a future stage
    count *unique* vehicles rather than re-counting the same car every frame.
    """

    track_id: Optional[int]
    class_name: str
    confidence: float
    bounding_box: tuple  # (x1, y1, x2, y2) in pixels
    frame_number: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VehicleTracker:
    """Assigns stable IDs to vehicles across frames using ByteTrack.

    Deliberately built as a wrapper AROUND a VehicleDetector rather than a
    parallel implementation:
      - reuses detector.model (no second YOLO model loaded into memory)
      - reuses detector.target_classes / confidence_threshold /
        iou_threshold / device (single source of truth stays in
        vehicle_detector.py / config.yaml)
      - reuses detector._id_to_name for class-id -> class-name mapping

    Usage:
        detector = VehicleDetector(...)
        tracker = VehicleTracker(detector, tracker_config="bytetrack.yaml")
        for frame_number, frame in enumerate(frames, start=1):
            tracked = tracker.track(frame, frame_number)
    """

    def __init__(
        self,
        detector: VehicleDetector,
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.detector = detector
        self.tracker_config = tracker_config
        # Fall back to the detector's own thresholds/device unless this
        # phase explicitly overrides them (mirrors config.yaml's
        # tracking.* : null -> inherit from detection semantics).
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else detector.confidence_threshold
        )
        self.iou_threshold = iou_threshold if iou_threshold is not None else detector.iou_threshold
        self.device = device if device is not None else detector.device

    def track(self, frame: np.ndarray, frame_number: int) -> List[TrackedVehicle]:
        """Run detection + ByteTrack association on a single BGR frame.

        `persist=True` tells Ultralytics to keep the tracker's internal
        state (active tracks, ByteTrack's track/lost buffers) alive between
        calls, so calling this once per video frame, in order, produces a
        continuous track history. Calling it on unrelated/out-of-order
        frames will corrupt tracking - see reset() to start a fresh video.
        """
        results = self.detector.model.track(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=list(self.detector.target_classes.values()),
            device=self.device,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )

        tracked: List[TrackedVehicle] = []
        if not results:
            return tracked

        result = results[0]
        if result.boxes is None:
            return tracked

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = self.detector._id_to_name.get(cls_id)
            if class_name is None:
                continue

            # box.id is None when ByteTrack has not (yet) confirmed an
            # identity for this detection (e.g. first frame it appears, or
            # a low-confidence "lost" candidate). We still report the
            # vehicle - just with track_id=None - rather than dropping it,
            # per "stable track ID where possible".
            track_id = int(box.id[0]) if box.id is not None else None

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            tracked.append(
                TrackedVehicle(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=(x1, y1, x2, y2),
                    frame_number=frame_number,
                )
            )
        return tracked

    def reset(self) -> None:
        """Clear ByteTrack's internal state (active/lost track buffers).

        Call this when switching to a new video or junction feed, otherwise
        the first frame of the new video would be associated against
        tracks left over from the previous one.
        """
        predictor = getattr(self.detector.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for t in predictor.trackers:
                t.reset()

    @staticmethod
    def unique_track_ids(tracked_vehicles: List[TrackedVehicle]) -> List[int]:
        """Convenience helper: distinct, known track_ids in a batch of
        TrackedVehicle objects (across one or more frames). Vehicles with
        track_id=None are excluded since they have no stable identity yet.

        This is the kind of helper the counting module can build on to
        count unique vehicles instead of re-counting the same car every
        frame - see run_tracking.py's summary output for a usage example.
        """
        seen = {tv.track_id for tv in tracked_vehicles if tv.track_id is not None}
        return sorted(seen)