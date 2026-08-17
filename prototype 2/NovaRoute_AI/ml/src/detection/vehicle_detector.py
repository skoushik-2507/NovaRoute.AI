"""
ml/src/detection/vehicle_detector.py

Phase 1 - YOLO vehicle detection.

Wraps a pretrained Ultralytics YOLO model and filters its output down to
road vehicle classes (car, motorcycle, bus, truck). No custom training,
no tracking, no congestion/risk logic - this module's only job is:
frame in -> list of Detection objects out.

This is imported directly by the tracking module later
(ml/src/tracking/vehicle_tracker.py) so the class-id mapping and detection
logic are defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Pretrained YOLO (COCO) class ids for the vehicle types this project cares
# about. These already exist in every stock YOLOv8 checkpoint, so no custom
# training is required.
DEFAULT_TARGET_CLASSES: Dict[str, int] = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


@dataclass
class Detection:
    """Structured result for a single detected vehicle."""

    class_id: int
    class_name: str
    confidence: float
    bounding_box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    track_id: Optional[int] = field(default=None)  # left None here; set later by tracker


class VehicleDetector:
    """Thin, configurable wrapper around a pretrained Ultralytics YOLO model.

    All behaviour (model weights, thresholds, target classes, device) is
    driven by arguments so it can be constructed straight from
    ml/config/config.yaml, with no hardcoded values in this file.
    """

    def __init__(
        self,
        model_weights: str = "yolov8n.pt",
        target_classes: Optional[Dict[str, int]] = None,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLO inference. Install project dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        self.model = YOLO(model_weights)
        self.target_classes: Dict[str, int] = target_classes or DEFAULT_TARGET_CLASSES
        self._id_to_name: Dict[int, str] = {v: k for k, v in self.target_classes.items()}
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame (image or video frame).

        Returns only the configured vehicle classes, each with class_id,
        class_name, confidence, and bounding_box already filled in.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=list(self.target_classes.values()),
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = self._id_to_name.get(cls_id)
            if class_name is None:
                # Shouldn't happen since we already filtered by `classes=`,
                # but guards against an unexpected id sneaking through.
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=(x1, y1, x2, y2),
                )
            )
        return detections

    @staticmethod
    def count_by_class(detections: List[Detection]) -> Dict[str, int]:
        """Aggregate a list of Detections into per-class counts + total.

        e.g. {'car': 27, 'bus': 3, 'truck': 5, 'motorcycle': 18, 'total': 53}
        """
        counts: Dict[str, int] = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        counts["total"] = sum(counts.values())
        return counts