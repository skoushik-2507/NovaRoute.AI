"""
tests/test_risk.py

Unit tests for ml/src/risk/risk_scorer.py.

Pure-python, no YOLO/ByteTrack/OpenCV/OSMnx dependency - RiskScorer only
does arithmetic on numbers callers hand it, so these run fast and fully
offline.

Run with: pytest tests/test_risk.py -v
"""

from datetime import datetime

import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from src.risk.risk_scorer import (  # noqa: E402
    RiskConfig,
    RiskScorer,
    RiskLevel,
    RiskResult,
    build_risk_config,
)


def make_scorer(**overrides) -> RiskScorer:
    return RiskScorer(RiskConfig(**overrides))


# --- low-risk junction ------------------------------------------------------


def test_low_risk_junction_is_classified_low():
    scorer = make_scorer()
    result = scorer.score(
        junction_id="quiet_lane",
        accident_count=0,
        traffic_density=0.1,
        pedestrian_count=2,
        hour=3,  # 3am, low risk hour in default profile
    )
    assert result.risk_level == RiskLevel.LOW
    assert result.overall_risk_score < 30.0
    assert result.factor_scores.accident_history == 0.0


# --- high-risk junction ------------------------------------------------------


def test_high_risk_junction_is_classified_critical():
    scorer = make_scorer()
    result = scorer.score(
        junction_id="busy_chowk",
        accident_count=15,      # above cap -> normalizes to 1.0
        traffic_density=2.0,    # above cap -> normalizes to 1.0
        pedestrian_count=80,    # above cap -> normalizes to 1.0
        hour=8,                 # morning rush, high risk hour in default profile
    )
    assert result.risk_level == RiskLevel.CRITICAL
    assert result.factor_scores.accident_history == 1.0
    assert result.factor_scores.traffic_density == 1.0
    assert result.factor_scores.pedestrian_conflict == 1.0
    assert result.overall_risk_score > 75.0


def test_high_risk_result_is_explainable_and_sums_correctly():
    scorer = make_scorer()
    result = scorer.score("busy_chowk", 15, 2.0, 80, hour=8)
    c = result.contributions
    total = c.accident_history + c.traffic_density + c.pedestrian_conflict + c.time_of_day
    assert total == pytest.approx(result.overall_risk_score)
    assert "Final risk score" in result.explain()


# --- boundary values ----------------------------------------------------


def test_zero_inputs_produce_zero_or_minimal_score():
    scorer = make_scorer()
    result = scorer.score("empty_road", accident_count=0, traffic_density=0.0, pedestrian_count=0, hour=2)
    assert result.factor_scores.accident_history == 0.0
    assert result.factor_scores.traffic_density == 0.0
    assert result.factor_scores.pedestrian_conflict == 0.0
    # time_of_day factor can still be > 0 (hour 2 has some baseline risk)
    assert result.overall_risk_score >= 0.0


def test_value_exactly_at_cap_normalizes_to_one():
    scorer = make_scorer(max_accidents_for_max_risk=10.0)
    result = scorer.score("r", accident_count=10, traffic_density=0.0, pedestrian_count=0, hour=0)
    assert result.factor_scores.accident_history == 1.0


def test_value_above_cap_is_clamped_not_exceeded():
    scorer = make_scorer(max_accidents_for_max_risk=10.0)
    result = scorer.score("r", accident_count=1000, traffic_density=0.0, pedestrian_count=0, hour=0)
    assert result.factor_scores.accident_history == 1.0  # clamped, not 100.0


def test_hour_boundaries_0_and_23_are_valid():
    scorer = make_scorer()
    r0 = scorer.score("r", 0, 0.0, 0, hour=0)
    r23 = scorer.score("r", 0, 0.0, 0, hour=23)
    assert isinstance(r0, RiskResult)
    assert isinstance(r23, RiskResult)


def test_risk_level_threshold_boundaries():
    # thresholds: low_max=30, moderate_max=55, high_max=75
    scorer = make_scorer(low_max=30.0, moderate_max=55.0, high_max=75.0)
    assert scorer._classify(30.0) == RiskLevel.LOW
    assert scorer._classify(30.01) == RiskLevel.MODERATE
    assert scorer._classify(55.0) == RiskLevel.MODERATE
    assert scorer._classify(55.01) == RiskLevel.HIGH
    assert scorer._classify(75.0) == RiskLevel.HIGH
    assert scorer._classify(75.01) == RiskLevel.CRITICAL


def test_weights_that_do_not_sum_to_one_are_normalized():
    scorer = make_scorer(
        weight_accident_history=1.0,
        weight_traffic_density=1.0,
        weight_pedestrian_conflict=1.0,
        weight_time_of_day=1.0,
    )
    # all equal -> normalized to 0.25 each
    assert scorer._weights["accident_history"] == pytest.approx(0.25)
    assert sum(scorer._weights.values()) == pytest.approx(1.0)


# --- invalid input ------------------------------------------------------


def test_negative_accident_count_raises():
    scorer = make_scorer()
    with pytest.raises(ValueError):
        scorer.score("r", accident_count=-1, traffic_density=0.5, pedestrian_count=5, hour=10)


def test_negative_traffic_density_raises():
    scorer = make_scorer()
    with pytest.raises(ValueError):
        scorer.score("r", accident_count=1, traffic_density=-0.1, pedestrian_count=5, hour=10)


def test_negative_pedestrian_count_raises():
    scorer = make_scorer()
    with pytest.raises(ValueError):
        scorer.score("r", accident_count=1, traffic_density=0.5, pedestrian_count=-5, hour=10)


def test_hour_out_of_range_raises():
    scorer = make_scorer()
    with pytest.raises(ValueError):
        scorer.score("r", accident_count=1, traffic_density=0.5, pedestrian_count=5, hour=24)
    with pytest.raises(ValueError):
        scorer.score("r", accident_count=1, traffic_density=0.5, pedestrian_count=5, hour=-1)


def test_non_numeric_accident_count_raises_type_error():
    scorer = make_scorer()
    with pytest.raises(TypeError):
        scorer.score("r", accident_count="lots", traffic_density=0.5, pedestrian_count=5, hour=10)


def test_bool_hour_raises_type_error():
    scorer = make_scorer()
    with pytest.raises(TypeError):
        scorer.score("r", accident_count=1, traffic_density=0.5, pedestrian_count=5, hour=True)


def test_invalid_traffic_density_type_raises():
    scorer = make_scorer()
    with pytest.raises(TypeError):
        scorer.score("r", accident_count=1, traffic_density=object(), pedestrian_count=5, hour=10)


def test_all_zero_weights_raises_on_construction():
    with pytest.raises(ValueError):
        RiskScorer(RiskConfig(
            weight_accident_history=0.0,
            weight_traffic_density=0.0,
            weight_pedestrian_conflict=0.0,
            weight_time_of_day=0.0,
        ))


def test_negative_weight_raises_on_construction():
    with pytest.raises(ValueError):
        RiskScorer(RiskConfig(weight_accident_history=-0.1))


# --- traffic_density duck-typed input shapes --------------------------------


class FakeCongestionResult:
    """Stands in for congestion.CongestionResult - only needs
    .traffic_density, exactly what _resolve_density() reads."""

    def __init__(self, traffic_density):
        self.traffic_density = traffic_density


def test_traffic_density_accepts_plain_float():
    scorer = make_scorer()
    result = scorer.score("r", 1, 0.6, 5, hour=10)
    assert result.factor_scores.traffic_density == pytest.approx(0.6 / scorer.config.max_density_for_max_risk)


def test_traffic_density_accepts_dict():
    scorer = make_scorer()
    result = scorer.score("r", 1, {"traffic_density": 0.6}, 5, hour=10)
    assert result.factor_scores.traffic_density == pytest.approx(0.6 / scorer.config.max_density_for_max_risk)


def test_traffic_density_accepts_object_with_attribute():
    scorer = make_scorer()
    result = scorer.score("r", 1, FakeCongestionResult(traffic_density=0.6), 5, hour=10)
    assert result.factor_scores.traffic_density == pytest.approx(0.6 / scorer.config.max_density_for_max_risk)


def test_hour_accepts_datetime():
    scorer = make_scorer()
    result = scorer.score("r", 1, 0.5, 5, hour=datetime(2026, 8, 16, 18, 30))
    expected = scorer.config.time_of_day_risk_profile[18]
    assert result.factor_scores.time_of_day == expected


# --- to_dict() / serialisation ----------------------------------------------


def test_result_to_dict_serialises_risk_level_as_string():
    scorer = make_scorer()
    result = scorer.score("junction_1", 3, 0.5, 10, hour=12)
    d = result.to_dict()
    assert isinstance(d["risk_level"], str)
    assert d["junction_id"] == "junction_1"
    assert set(d["contributions"].keys()) == {
        "accident_history", "traffic_density", "pedestrian_conflict", "time_of_day"
    }


# --- build_risk_config() ----------------------------------------------------


def test_build_risk_config_reads_full_section():
    cfg = {
        "risk": {
            "weights": {
                "accident_history": 0.4,
                "traffic_density": 0.3,
                "pedestrian_conflict": 0.2,
                "time_of_day": 0.1,
            },
            "normalization": {
                "max_accidents_for_max_risk": 20.0,
                "max_density_for_max_risk": 2.0,
                "max_pedestrian_for_max_risk": 100.0,
            },
            "risk_level_thresholds": {
                "low_max": 25.0,
                "moderate_max": 50.0,
                "high_max": 70.0,
            },
        }
    }
    config = build_risk_config(cfg)
    assert config.weight_accident_history == 0.4
    assert config.max_accidents_for_max_risk == 20.0
    assert config.low_max == 25.0


def test_build_risk_config_falls_back_to_defaults_when_section_missing():
    config = build_risk_config({})
    assert config.weight_accident_history == RiskConfig.weight_accident_history
    assert config.max_accidents_for_max_risk == RiskConfig.max_accidents_for_max_risk
    assert len(config.time_of_day_risk_profile) == 24