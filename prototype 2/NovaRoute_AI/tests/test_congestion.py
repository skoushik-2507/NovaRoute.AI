"""
tests/test_congestion.py

Unit tests for ml/src/traffic/congestion.py.

Pure-python, no YOLO/ByteTrack/OpenCV dependency - CongestionEstimator only
does arithmetic on numbers callers hand it, so these tests run fast and
fully offline.

Run with: pytest tests/test_congestion.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from src.traffic.congestion import (  # noqa: E402
    CongestionConfig,
    CongestionEstimator,
    CongestionLevel,
    CongestionResult,
    build_congestion_config,
)


def make_estimator(**overrides) -> CongestionEstimator:
    """Default CongestionConfig (matches CongestionConfig's own defaults,
    which mirror config.yaml's `congestion` section) with optional
    per-test overrides."""
    return CongestionEstimator(CongestionConfig(**overrides))


# --- low / medium / high traffic scenarios --------------------------------
# capacity=50 throughout; total_count represents average active vehicles in the window.


def test_low_traffic_is_free_flow_with_factor_near_one():
    estimator = make_estimator()
    result = estimator.estimate(
        road_id="junction_1",
        vehicle_counts=10,  # density = 10/50 = 0.2
        window_start=0.0,
        window_end=30.0,
        road_capacity=50.0,
    )

    assert result.total_vehicle_count == 10
    assert result.traffic_density == 0.2
    assert result.congestion_level == CongestionLevel.FREE_FLOW
    # BPR: 1 + 0.15 * 0.2**4 = 1.00024 - very close to free-flow (1.0).
    assert result.congestion_factor == 1.0 + 0.15 * (0.2**4)
    assert 1.0 <= result.congestion_factor < 1.01


def test_medium_traffic_is_moderate_with_higher_factor():
    estimator = make_estimator()
    result = estimator.estimate(
        road_id="junction_1",
        vehicle_counts=35,  # density = 35/50 = 0.7
        window_start=0.0,
        window_end=30.0,
        road_capacity=50.0,
    )

    assert result.traffic_density == 0.7
    assert result.congestion_level == CongestionLevel.MODERATE
    assert result.congestion_factor == 1.0 + 0.15 * (0.7**4)
    assert 1.02 < result.congestion_factor < 1.1


def test_high_traffic_is_heavy_with_significantly_higher_factor():
    estimator = make_estimator()
    result = estimator.estimate(
        road_id="junction_1",
        vehicle_counts=55,  # density = 55/50 = 1.1
        window_start=0.0,
        window_end=30.0,
        road_capacity=50.0,
    )

    assert result.traffic_density == 1.1
    assert result.congestion_level == CongestionLevel.HEAVY
    assert result.congestion_factor == 1.0 + 0.15 * (1.1**4)
    assert result.congestion_factor > 1.2


def test_severe_traffic_over_capacity_is_classified_severe():
    estimator = make_estimator()
    result = estimator.estimate(
        road_id="junction_1",
        vehicle_counts=65,  # density = 65/50 = 1.3 > heavy_max (1.2)
        window_start=0.0,
        window_end=30.0,
        road_capacity=50.0,
    )

    assert result.congestion_level == CongestionLevel.SEVERE
    assert result.congestion_factor > 1.4


def test_congestion_factor_increases_monotonically_with_density():
    estimator = make_estimator()
    low = estimator.estimate("r", 10, 0, 30, road_capacity=50.0)
    medium = estimator.estimate("r", 35, 0, 30, road_capacity=50.0)
    high = estimator.estimate("r", 55, 0, 30, road_capacity=50.0)
    severe = estimator.estimate("r", 65, 0, 30, road_capacity=50.0)

    assert low.congestion_factor < medium.congestion_factor < high.congestion_factor < severe.congestion_factor


# --- capacity resolution ---------------------------------------------------


def test_explicit_road_capacity_overrides_config_defaults():
    estimator = make_estimator(default_capacity=50.0, default_capacity_by_road={"junction_1": 80.0})
    result = estimator.estimate("junction_1", vehicle_counts=20, window_start=0, window_end=10, road_capacity=200.0)
    assert result.road_capacity == 200.0
    assert result.traffic_density == 20 / 200.0


def test_per_road_capacity_used_when_no_explicit_override():
    estimator = make_estimator(default_capacity=50.0, default_capacity_by_road={"junction_1": 80.0})
    result = estimator.estimate("junction_1", vehicle_counts=20, window_start=0, window_end=10)
    assert result.road_capacity == 80.0


def test_global_default_capacity_used_when_no_per_road_entry():
    estimator = make_estimator(default_capacity=50.0, default_capacity_by_road={"junction_1": 80.0})
    result = estimator.estimate("junction_99", vehicle_counts=20, window_start=0, window_end=10)
    assert result.road_capacity == 50.0


def test_zero_or_negative_road_capacity_raises():
    estimator = make_estimator()
    for bad_capacity in (0, -5):
        try:
            estimator.estimate("r", 10, 0, 10, road_capacity=bad_capacity)
            assert False, "expected ValueError"
        except ValueError:
            pass


# --- vehicle_counts input shapes (decoupled from counting/tracking modules) -


class FakeWindowCount:
    """Stands in for vehicle_counter.WindowVehicleCount - only needs
    `.total`, exactly what _resolve_total_count() reads."""

    def __init__(self, total):
        self.total = total


def test_vehicle_counts_accepts_plain_int():
    estimator = make_estimator()
    result = estimator.estimate("r", 42, 0, 10, road_capacity=100)
    assert result.total_vehicle_count == 42


def test_vehicle_counts_accepts_dict_with_total_key():
    estimator = make_estimator()
    result = estimator.estimate("r", {"car": 10, "bus": 2, "total": 12}, 0, 10, road_capacity=100)
    assert result.total_vehicle_count == 12


def test_vehicle_counts_accepts_dict_without_total_key_sums_values():
    estimator = make_estimator()
    result = estimator.estimate("r", {"car": 10, "bus": 2, "truck": 3}, 0, 10, road_capacity=100)
    assert result.total_vehicle_count == 15


def test_vehicle_counts_accepts_object_with_total_attribute():
    estimator = make_estimator()
    result = estimator.estimate("r", FakeWindowCount(total=7), 0, 10, road_capacity=100)
    assert result.total_vehicle_count == 7


def test_vehicle_counts_rejects_unsupported_type():
    estimator = make_estimator()
    try:
        estimator.estimate("r", object(), 0, 10, road_capacity=100)
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_vehicle_counts_rejects_bool():
    estimator = make_estimator()
    try:
        estimator.estimate("r", True, 0, 10, road_capacity=100)
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_negative_vehicle_count_raises():
    estimator = make_estimator()
    try:
        estimator.estimate("r", -1, 0, 10, road_capacity=100)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- window validation -------------------------------------------------


def test_window_end_before_window_start_raises():
    estimator = make_estimator()
    try:
        estimator.estimate("r", 10, window_start=30, window_end=0, road_capacity=100)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- clamp -----------------------------------------------------------------


def test_congestion_factor_is_clamped_to_configured_max():
    estimator = make_estimator(max_congestion_factor=5.0)
    result = estimator.estimate("r", vehicle_counts=1000, window_start=0, window_end=10, road_capacity=1)
    assert result.congestion_factor == 5.0


# --- dynamic_travel_time ----------------------------------------------------


def test_dynamic_travel_time_multiplies_base_by_factor():
    result = CongestionResult(
        road_id="r",
        window_start=0,
        window_end=10,
        total_vehicle_count=10,
        road_capacity=50.0,
        traffic_density=0.2,
        congestion_level=CongestionLevel.FREE_FLOW,
        congestion_factor=2.0,
    )
    assert CongestionEstimator.dynamic_travel_time(result, base_travel_time=30.0) == 60.0


def test_dynamic_travel_time_rejects_negative_base_time():
    result = CongestionResult(
        road_id="r",
        window_start=0,
        window_end=10,
        total_vehicle_count=10,
        road_capacity=50.0,
        traffic_density=0.2,
        congestion_level=CongestionLevel.FREE_FLOW,
        congestion_factor=2.0,
    )
    try:
        CongestionEstimator.dynamic_travel_time(result, base_travel_time=-1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- to_dict() / serialisation ----------------------------------------------


def test_result_to_dict_serialises_congestion_level_as_string():
    estimator = make_estimator()
    result = estimator.estimate("junction_1", 10, 0, 30, road_capacity=50.0)
    d = result.to_dict()
    assert d["congestion_level"] == "free_flow"
    assert d["road_id"] == "junction_1"
    assert isinstance(d["congestion_factor"], float)


# --- build_congestion_config() ----------------------------------------------


def test_build_congestion_config_reads_full_section():
    cfg = {
        "congestion": {
            "alpha": 0.2,
            "beta": 3.0,
            "default_capacity": 60.0,
            "default_capacity_by_road": {"junction_5": 90.0},
            "free_flow_max": 0.4,
            "moderate_max": 0.75,
            "heavy_max": 1.1,
            "max_congestion_factor": 6.0,
        }
    }
    config = build_congestion_config(cfg)
    assert config.alpha == 0.2
    assert config.beta == 3.0
    assert config.default_capacity == 60.0
    assert config.default_capacity_by_road == {"junction_5": 90.0}
    assert config.free_flow_max == 0.4
    assert config.moderate_max == 0.75
    assert config.heavy_max == 1.1
    assert config.max_congestion_factor == 6.0


def test_build_congestion_config_falls_back_to_defaults_when_section_missing():
    config = build_congestion_config({})
    assert config.alpha == CongestionConfig.alpha
    assert config.beta == CongestionConfig.beta
    assert config.default_capacity == CongestionConfig.default_capacity
    assert config.default_capacity_by_road == {}