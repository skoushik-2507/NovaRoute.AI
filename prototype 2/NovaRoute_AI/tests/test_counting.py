import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from src.traffic.vehicle_counter import VehicleCounter, WindowVehicleCount, VEHICLE_CLASSES


class FakeDetection:
    def __init__(self, class_name: str):
        self.class_name = class_name


def test_empty_detections_returns_all_zero():
    result = VehicleCounter.count_frame([], frame_index=1)
    assert result.to_dict() == {
        "frame_index": 1, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0
    }


def test_single_vehicle_is_counted():
    result = VehicleCounter.count_frame([FakeDetection("car")], frame_index=5)
    assert result.car == 1 and result.total == 1


def test_multiple_vehicle_classes():
    result = VehicleCounter.count_frame([
        FakeDetection("car"), FakeDetection("car"), FakeDetection("bus"),
        FakeDetection("truck"), FakeDetection("motorcycle"), FakeDetection("motorcycle"),
    ], frame_index=10)
    assert result.car == 2 and result.bus == 1 and result.truck == 1 and result.motorcycle == 2
    assert result.total == 6


def test_unknown_classes_are_ignored():
    result = VehicleCounter.count_frame([FakeDetection("car"), FakeDetection("person")])
    assert result.car == 1 and result.total == 1


def test_window_uses_average_not_sum():
    counter = VehicleCounter(window_size=3)
    counter.update([FakeDetection("car"), FakeDetection("car")])
    counter.update([FakeDetection("car"), FakeDetection("car")])
    counter.update([FakeDetection("car")])
    window = counter.get_window_aggregate()
    assert isinstance(window, WindowVehicleCount)
    assert window.total == 2
    assert window.avg_total_per_frame == pytest.approx(5 / 3)
    assert window.peak_total == 2
    assert window.aggregation_method == "average_active_vehicles"


def test_window_drops_oldest_frame():
    counter = VehicleCounter(window_size=2)
    counter.update([FakeDetection("car")])
    counter.update([FakeDetection("bus"), FakeDetection("bus")])
    counter.update([FakeDetection("truck")])
    window = counter.get_window_aggregate()
    assert window.start_frame == 2 and window.end_frame == 3
    assert window.total == 2
    assert window.car == 0 and window.bus == 1 and window.truck == 1


def test_reset_clears_state():
    counter = VehicleCounter(window_size=5)
    counter.update([FakeDetection("car")])
    counter.reset()
    assert counter.frame_index == 0
    assert counter.get_window_aggregate().frame_count == 0


def test_invalid_window_size_raises():
    with pytest.raises(ValueError):
        VehicleCounter(window_size=0)
    with pytest.raises(ValueError):
        VehicleCounter(window_size=True)


def test_vehicle_classes_constant():
    assert set(VEHICLE_CLASSES) == {"car", "motorcycle", "bus", "truck"}
