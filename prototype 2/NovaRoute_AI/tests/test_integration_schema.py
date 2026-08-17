import json
from pathlib import Path

import jsonschema


def test_traffic_schema_is_valid_and_accepts_pipeline_shape():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "integration/schemas/traffic_data.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    sample = {
        "schema_version": "1.1.0",
        "road_segment_id": "junction_1",
        "osm_edge": None,
        "timestamp": "2026-08-16T12:00:00+00:00",
        "observation_window_seconds": 1.2,
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
            "accident_history": 0.3,
            "traffic_density": 0.3733333333,
            "pedestrian_conflict": 0.2,
            "time_of_day": 0.75,
        },
        "risk_contributions": {
            "accident_history": 10.5,
            "traffic_density": 9.3333333333,
            "pedestrian_conflict": 5.0,
            "time_of_day": 11.25,
        },
    }
    jsonschema.Draft202012Validator(schema).validate(sample)
