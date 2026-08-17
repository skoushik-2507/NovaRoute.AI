"""Shared OpenCV drawing helpers for detection/tracking scripts."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

# Distinct BGR colour per vehicle class, for consistent visuals across scripts
CLASS_COLORS = {
    "car": (60, 180, 75),
    "bus": (0, 130, 200),
    "truck": (245, 130, 48),
    "motorcycle": (145, 30, 180),
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(frame: np.ndarray, detections: List) -> np.ndarray:
    """Draw bounding boxes + class label for each Detection (no track id)."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bounding_box]
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        _draw_label(frame, label, x1, y1, color)
    return frame


def draw_tracks(frame: np.ndarray, tracked_vehicles: List) -> np.ndarray:
    """Draw bounding boxes + class label + persistent track id."""
    for tv in tracked_vehicles:
        x1, y1, x2, y2 = [int(v) for v in tv.bounding_box]
        color = CLASS_COLORS.get(tv.class_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        id_str = f"#{tv.track_id}" if tv.track_id is not None else "?"
        label = f"{tv.class_name} {id_str}"
        _draw_label(frame, label, x1, y1, color)
    return frame


def put_summary_text(frame: np.ndarray, lines: List[str], origin=(10, 25)) -> np.ndarray:
    """Draw a small multi-line summary box (e.g. per-class counts) in the
    top-left corner of the frame."""
    x, y = origin
    line_height = 22
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return frame


def _draw_label(frame: np.ndarray, label: str, x1: int, y1: int, color) -> None:
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )