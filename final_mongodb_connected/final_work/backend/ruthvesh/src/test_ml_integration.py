"""
test_ml_integration.py

Pytest test suite for src/ml_integration.py (the ML-to-routing adapter).

These tests do not touch the graph, routing.py, or congestion.py's
congestion-factor calculation - the adapter's whole job is to read an
already-computed ML congestion_factor and map it to explicit, verified
OSM segments, so these tests exercise exactly that: loading/validation,
junction->segment mapping, and building the segment congestion map,
including the required failure modes (unknown junction, malformed JSON,
conflicting segment assignment).

Run with:
    pytest src/test_ml_integration.py -v
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import ml_integration as mi
from src.congestion import get_segment_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_observation():
    """A minimal, schema-valid ML observation (mirrors traffic_data.json's
    own example, trimmed to the fields this adapter reads plus the
    required-but-unused ones a real file would also carry)."""
    return {
        "schema_version": "1.1.0",
        "road_segment_id": "junction_1",
        "osm_edge": None,
        "timestamp": "2026-08-16T12:00:00+00:00",
        "observation_window_seconds": 5.0,
        "vehicle_counts": {"car": 18, "motorcycle": 7, "bus": 1, "truck": 2, "total": 28},
        "total_vehicles": 28,
        "peak_vehicles": 33,
        "road_capacity": 50.0,
        "traffic_density": 0.56,
        "congestion_level": "moderate",
        "congestion_factor": 1.01475,
        "aggregation_method": "average_active_vehicles",
        "risk_score": 49.25,
        "risk_level": "moderate",
        "risk_factor_scores": {
            "accident_history": 0.3, "traffic_density": 0.373333,
            "pedestrian_conflict": 0.2, "time_of_day": 0.75,
        },
        "risk_contributions": {
            "accident_history": 10.5, "traffic_density": 9.333333,
            "pedestrian_conflict": 5.0, "time_of_day": 11.25,
        },
    }


@pytest.fixture
def sample_mapping():
    """A fake-but-well-formed JUNCTION_TO_SEGMENTS override, so tests
    never depend on the real (currently empty) module-level mapping."""
    return {
        "junction_1": [(111, 222, 0), (222, 111, 0)],
        "junction_2": [(333, 444, 0)],
    }


# ---------------------------------------------------------------------------
# validate_ml_observation / load_ml_observation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_observation_passes(self, valid_observation):
        mi.validate_ml_observation(valid_observation)  # should not raise

    def test_non_dict_rejected(self):
        with pytest.raises(mi.MLObservationError, match="must be a JSON object"):
            mi.validate_ml_observation(["not", "a", "dict"])

    def test_missing_schema_version_rejected(self, valid_observation):
        del valid_observation["schema_version"]
        with pytest.raises(mi.MLObservationError, match="schema_version"):
            mi.validate_ml_observation(valid_observation)

    def test_unsupported_schema_version_rejected(self, valid_observation):
        valid_observation["schema_version"] = "2.0.0"
        with pytest.raises(mi.MLObservationError, match="Unsupported ML observation schema_version"):
            mi.validate_ml_observation(valid_observation)

    def test_empty_road_segment_id_rejected(self, valid_observation):
        valid_observation["road_segment_id"] = "   "
        with pytest.raises(mi.MLObservationError, match="road_segment_id"):
            mi.validate_ml_observation(valid_observation)

    def test_bad_timestamp_rejected(self, valid_observation):
        valid_observation["timestamp"] = "not-a-timestamp"
        with pytest.raises(mi.MLObservationError, match="timestamp"):
            mi.validate_ml_observation(valid_observation)

    def test_congestion_factor_below_one_rejected(self, valid_observation):
        valid_observation["congestion_factor"] = 0.5
        with pytest.raises(mi.MLObservationError, match="congestion_factor"):
            mi.validate_ml_observation(valid_observation)

    def test_congestion_factor_non_numeric_rejected(self, valid_observation):
        valid_observation["congestion_factor"] = "high"
        with pytest.raises(mi.MLObservationError, match="congestion_factor"):
            mi.validate_ml_observation(valid_observation)

    def test_risk_score_out_of_range_rejected(self, valid_observation):
        valid_observation["risk_score"] = 150.0
        with pytest.raises(mi.MLObservationError, match="risk_score"):
            mi.validate_ml_observation(valid_observation)

    def test_invalid_risk_level_rejected(self, valid_observation):
        valid_observation["risk_level"] = "catastrophic"
        with pytest.raises(mi.MLObservationError, match="risk_level"):
            mi.validate_ml_observation(valid_observation)

    def test_vehicle_counts_missing_key_rejected(self, valid_observation):
        del valid_observation["vehicle_counts"]["bus"]
        with pytest.raises(mi.MLObservationError, match="vehicle_counts"):
            mi.validate_ml_observation(valid_observation)

    def test_vehicle_counts_negative_rejected(self, valid_observation):
        valid_observation["vehicle_counts"]["car"] = -1
        with pytest.raises(mi.MLObservationError, match="vehicle_counts"):
            mi.validate_ml_observation(valid_observation)

    def test_vehicle_counts_non_integer_rejected(self, valid_observation):
        valid_observation["vehicle_counts"]["car"] = 3.5
        with pytest.raises(mi.MLObservationError, match="vehicle_counts"):
            mi.validate_ml_observation(valid_observation)


class TestLoadMlObservation:
    def test_load_from_dict(self, valid_observation):
        obs = mi.load_ml_observation(valid_observation)
        assert obs["road_segment_id"] == "junction_1"

    def test_load_from_file(self, tmp_path, valid_observation):
        path = tmp_path / "obs.json"
        path.write_text(json.dumps(valid_observation), encoding="utf-8")
        obs = mi.load_ml_observation(path)
        assert obs["congestion_factor"] == pytest.approx(1.01475)

    def test_load_from_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(mi.MLObservationError, match="not found"):
            mi.load_ml_observation(missing)

    def test_load_from_malformed_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(mi.MLObservationError, match="not valid JSON"):
            mi.load_ml_observation(path)

    def test_load_rejects_invalid_observation(self, valid_observation):
        del valid_observation["risk_level"]
        with pytest.raises(mi.MLObservationError):
            mi.load_ml_observation(valid_observation)


class TestExtractMlFields:
    def test_extracts_expected_fields(self, valid_observation):
        fields = mi.extract_ml_fields(valid_observation)
        assert fields.road_segment_id == "junction_1"
        assert fields.congestion_factor == pytest.approx(1.01475)
        assert fields.risk_score == pytest.approx(49.25)
        assert fields.risk_level == "moderate"
        assert fields.vehicle_counts["total"] == 28
        assert fields.schema_version == "1.1.0"

    def test_extract_validates_first(self, valid_observation):
        valid_observation["congestion_factor"] = 0.1
        with pytest.raises(mi.MLObservationError):
            mi.extract_ml_fields(valid_observation)


# ---------------------------------------------------------------------------
# map_junction_to_segments
# ---------------------------------------------------------------------------

class TestMapJunctionToSegments:
    def test_known_junction_returns_verified_segments(self, sample_mapping):
        segments = mi.map_junction_to_segments("junction_1", junction_to_segments=sample_mapping)
        assert segments == [(111, 222, 0), (222, 111, 0)]

    def test_unknown_junction_raises_clear_error(self, sample_mapping):
        with pytest.raises(mi.UnknownJunctionError, match="junction_999"):
            mi.map_junction_to_segments("junction_999", junction_to_segments=sample_mapping)

    def test_module_level_mapping_is_empty_by_default(self):
        # This is the current, correct state: no junction has been
        # verified yet, so every lookup against the real module-level
        # JUNCTION_TO_SEGMENTS must fail rather than guess.
        assert mi.JUNCTION_TO_SEGMENTS == {}
        with pytest.raises(mi.UnknownJunctionError):
            mi.map_junction_to_segments("junction_1")

    def test_malformed_segment_tuple_rejected(self):
        bad_mapping = {"junction_1": [(111, 222)]}  # missing key
        with pytest.raises(mi.MLObservationError, match=r"\(u, v, key\)"):
            mi.map_junction_to_segments("junction_1", junction_to_segments=bad_mapping)

    def test_explicitly_empty_segment_list_is_not_an_error(self):
        # An explicit empty list means "verified: zero segments so far",
        # which is different from the key being absent entirely.
        mapping = {"junction_1": []}
        assert mi.map_junction_to_segments("junction_1", junction_to_segments=mapping) == []


# ---------------------------------------------------------------------------
# build_segment_congestion_map
# ---------------------------------------------------------------------------

class TestBuildSegmentCongestionMap:
    def test_builds_expected_map(self, valid_observation, sample_mapping):
        result = mi.build_segment_congestion_map([valid_observation], junction_to_segments=sample_mapping)
        expected_ids = {get_segment_id(111, 222, 0), get_segment_id(222, 111, 0)}
        assert set(result.keys()) == expected_ids
        for seg_id in expected_ids:
            assert result[seg_id] == pytest.approx(1.01475)

    def test_uses_ml_factor_verbatim_not_recomputed(self, valid_observation, sample_mapping):
        # The ML congestion_factor (1.01475) does NOT match what
        # congestion.calculate_congestion_factor() would produce from
        # vehicle_counts under ruthvesh's own default config - this
        # test guards against ever accidentally re-deriving the factor
        # instead of reading it straight from the ML observation.
        from src.congestion import calculate_congestion_factor
        recomputed = calculate_congestion_factor(valid_observation["vehicle_counts"]["total"])
        assert recomputed != pytest.approx(valid_observation["congestion_factor"])

        result = mi.build_segment_congestion_map([valid_observation], junction_to_segments=sample_mapping)
        seg_id = get_segment_id(111, 222, 0)
        assert result[seg_id] == pytest.approx(valid_observation["congestion_factor"])
        assert result[seg_id] != pytest.approx(recomputed)

    def test_unknown_junction_raises_by_default(self, valid_observation):
        with pytest.raises(mi.UnknownJunctionError):
            mi.build_segment_congestion_map([valid_observation], junction_to_segments={})

    def test_unknown_junction_skipped_when_requested(self, valid_observation):
        result = mi.build_segment_congestion_map(
            [valid_observation], junction_to_segments={}, on_unknown_junction="skip"
        )
        assert result == {}

    def test_invalid_on_unknown_junction_value_raises(self, valid_observation, sample_mapping):
        with pytest.raises(ValueError, match="on_unknown_junction"):
            mi.build_segment_congestion_map(
                [valid_observation], junction_to_segments=sample_mapping,
                on_unknown_junction="ignore",
            )

    def test_multiple_observations_combine(self, valid_observation, sample_mapping):
        obs2 = copy.deepcopy(valid_observation)
        obs2["road_segment_id"] = "junction_2"
        obs2["congestion_factor"] = 2.5

        result = mi.build_segment_congestion_map([valid_observation, obs2], junction_to_segments=sample_mapping)

        assert result[get_segment_id(111, 222, 0)] == pytest.approx(1.01475)
        assert result[get_segment_id(333, 444, 0)] == pytest.approx(2.5)

    def test_conflicting_congestion_factor_on_shared_segment_raises(self, valid_observation, sample_mapping):
        overlapping_mapping = dict(sample_mapping)
        overlapping_mapping["junction_2"] = [(111, 222, 0)]  # same segment as junction_1

        obs2 = copy.deepcopy(valid_observation)
        obs2["road_segment_id"] = "junction_2"
        obs2["congestion_factor"] = 3.0  # different from junction_1's 1.01475

        with pytest.raises(mi.SegmentCongestionConflictError, match="junction_1.*junction_2|junction_2.*junction_1"):
            mi.build_segment_congestion_map([valid_observation, obs2], junction_to_segments=overlapping_mapping)

    def test_identical_congestion_factor_on_shared_segment_does_not_raise(self, valid_observation, sample_mapping):
        overlapping_mapping = dict(sample_mapping)
        overlapping_mapping["junction_2"] = [(111, 222, 0)]  # same segment as junction_1

        obs2 = copy.deepcopy(valid_observation)
        obs2["road_segment_id"] = "junction_2"
        obs2["congestion_factor"] = valid_observation["congestion_factor"]  # identical, no conflict

        result = mi.build_segment_congestion_map([valid_observation, obs2], junction_to_segments=overlapping_mapping)
        assert result[get_segment_id(111, 222, 0)] == pytest.approx(valid_observation["congestion_factor"])

    def test_empty_observations_returns_empty_map(self, sample_mapping):
        assert mi.build_segment_congestion_map([], junction_to_segments=sample_mapping) == {}