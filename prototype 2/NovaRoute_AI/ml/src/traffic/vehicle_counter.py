"""Vehicle counting utilities for NovaRoute.AI.

The counter deliberately distinguishes two concepts:

* per-frame detections: how many vehicles YOLO sees in one frame;
* active vehicles: how many tracked vehicles are present in a frame.

For congestion estimation we use the rolling *average active vehicles per
frame*. We do not sum detections across frames, because that would count the
same physical vehicle repeatedly and produce an invalid traffic measurement.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Sequence

VEHICLE_CLASSES: List[str] = ["car", "motorcycle", "bus", "truck"]


@dataclass
class FrameVehicleCount:
    frame_index: int
    car: int = 0
    motorcycle: int = 0
    bus: int = 0
    truck: int = 0
    total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WindowVehicleCount:
    """Rolling-window estimate of active vehicles.

    ``total`` and the class fields are rounded averages of active tracked
    vehicles per frame in the window. ``peak_total`` is the largest active
    count observed in the window. This is suitable for a concurrent
    vehicle-capacity / occupancy-style congestion model.
    """

    start_frame: int
    end_frame: int
    frame_count: int
    car: int = 0
    motorcycle: int = 0
    bus: int = 0
    truck: int = 0
    total: int = 0
    avg_total_per_frame: float = 0.0
    peak_total: int = 0
    aggregation_method: str = "average_active_vehicles"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VehicleCounter:
    """Count detections and tracked vehicles over a rolling frame window."""

    def __init__(self, window_size: int = 30):
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self._frame_counts: Deque[FrameVehicleCount] = deque(maxlen=window_size)
        self.frame_index = 0

    @staticmethod
    def count_frame(detections: Sequence, frame_index: int = 0) -> FrameVehicleCount:
        """Count vehicles in one frame from objects exposing ``class_name``."""
        counts = {cls_name: 0 for cls_name in VEHICLE_CLASSES}
        for det in detections:
            class_name = getattr(det, "class_name", None)
            if class_name in counts:
                counts[class_name] += 1
        total = sum(counts.values())
        return FrameVehicleCount(frame_index=frame_index, total=total, **counts)

    def update(self, detections: Sequence) -> FrameVehicleCount:
        """Record a raw per-frame detection count."""
        self.frame_index += 1
        frame_count = self.count_frame(detections, frame_index=self.frame_index)
        self._frame_counts.append(frame_count)
        return frame_count

    def update_tracked(self, tracked_vehicles: Sequence) -> FrameVehicleCount:
        """Record the active tracked vehicles in one frame.

        A vehicle without a ByteTrack ID is still included in the current
        occupancy estimate; it simply cannot contribute to unique-ID metrics.
        """
        return self.update(tracked_vehicles)

    def get_window_aggregate(self) -> WindowVehicleCount:
        """Return the average active vehicle count across the rolling window.

        This intentionally does NOT sum frame counts. Summing would count the
        same vehicle once per frame and would inflate congestion.
        """
        if not self._frame_counts:
            return WindowVehicleCount(start_frame=0, end_frame=0, frame_count=0)

        n = len(self._frame_counts)
        averages: Dict[str, float] = {
            cls_name: sum(getattr(fc, cls_name) for fc in self._frame_counts) / n
            for cls_name in VEHICLE_CLASSES
        }
        avg_total = sum(averages.values())
        peak_total = max(fc.total for fc in self._frame_counts)

        rounded_total = int(round(avg_total))
        # Allocate the rounded total across classes using largest remainders so
        # the class values always sum exactly to the reported total.
        rounded = {cls_name: int(value) for cls_name, value in averages.items()}
        remainder = rounded_total - sum(rounded.values())
        if remainder > 0:
            order = sorted(
                VEHICLE_CLASSES,
                key=lambda name: averages[name] - int(averages[name]),
                reverse=True,
            )
            for cls_name in order[:remainder]:
                rounded[cls_name] += 1

        return WindowVehicleCount(
            start_frame=self._frame_counts[0].frame_index,
            end_frame=self._frame_counts[-1].frame_index,
            frame_count=n,
            total=rounded_total,
            avg_total_per_frame=avg_total,
            peak_total=peak_total,
            **rounded,
        )

    def reset(self) -> None:
        self._frame_counts.clear()
        self.frame_index = 0
