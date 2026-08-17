"""Transparent congestion estimation for NovaRoute.AI.

The estimator consumes the rolling average number of concurrently active
vehicles and a calibrated concurrent vehicle capacity for a road segment.
The resulting volume/capacity ratio is fed into the BPR volume-delay
function:

    congestion_factor = 1 + alpha * (v/c ** beta)

The factor is the only value the Dijkstra layer needs to adjust free-flow
travel time:

    dynamic_travel_time = base_travel_time * congestion_factor
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Dict, Optional, Union


class CongestionLevel(str, Enum):
    FREE_FLOW = "free_flow"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"


@dataclass
class CongestionConfig:
    alpha: float = 0.15
    beta: float = 4.0
    default_capacity: float = 50.0
    default_capacity_by_road: Dict[str, float] = field(default_factory=dict)
    free_flow_max: float = 0.5
    moderate_max: float = 0.8
    heavy_max: float = 1.2
    max_congestion_factor: float = 8.0

    def __post_init__(self) -> None:
        numeric_positive = {
            "alpha": self.alpha,
            "beta": self.beta,
            "default_capacity": self.default_capacity,
            "max_congestion_factor": self.max_congestion_factor,
        }
        for name, value in numeric_positive.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a finite number > 0")
        if not (0 <= self.free_flow_max <= self.moderate_max <= self.heavy_max):
            raise ValueError("congestion thresholds must satisfy 0 <= free_flow <= moderate <= heavy")
        for road_id, capacity in self.default_capacity_by_road.items():
            if not isinstance(road_id, str) or not road_id.strip():
                raise ValueError("road ids in default_capacity_by_road must be non-empty strings")
            if isinstance(capacity, bool) or not isinstance(capacity, (int, float)) or not isfinite(float(capacity)) or float(capacity) <= 0:
                raise ValueError(f"capacity for {road_id!r} must be a finite number > 0")


def build_congestion_config(cfg: dict) -> CongestionConfig:
    section = cfg.get("congestion", {}) or {}
    return CongestionConfig(
        alpha=section.get("alpha", CongestionConfig.alpha),
        beta=section.get("beta", CongestionConfig.beta),
        default_capacity=section.get("default_capacity", CongestionConfig.default_capacity),
        default_capacity_by_road=dict(section.get("default_capacity_by_road", {}) or {}),
        free_flow_max=section.get("free_flow_max", CongestionConfig.free_flow_max),
        moderate_max=section.get("moderate_max", CongestionConfig.moderate_max),
        heavy_max=section.get("heavy_max", CongestionConfig.heavy_max),
        max_congestion_factor=section.get("max_congestion_factor", CongestionConfig.max_congestion_factor),
    )


@dataclass
class CongestionResult:
    road_id: str
    window_start: float
    window_end: float
    total_vehicle_count: int
    road_capacity: float
    traffic_density: float
    congestion_level: CongestionLevel
    congestion_factor: float
    aggregation_method: str = "average_active_vehicles"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["congestion_level"] = self.congestion_level.value
        return data


VehicleCountsInput = Union[int, float, Dict[str, Any], Any]


class CongestionEstimator:
    def __init__(self, config: Optional[CongestionConfig] = None):
        self.config = config or CongestionConfig()

    def estimate(
        self,
        road_id: str,
        vehicle_counts: VehicleCountsInput,
        window_start: float,
        window_end: float,
        road_capacity: Optional[float] = None,
        aggregation_method: str = "average_active_vehicles",
    ) -> CongestionResult:
        if not isinstance(road_id, str) or not road_id.strip():
            raise ValueError("road_id must be a non-empty string")
        if not isfinite(float(window_start)) or not isfinite(float(window_end)):
            raise ValueError("window_start and window_end must be finite")
        if window_end < window_start:
            raise ValueError("window_end must be >= window_start")
        if not aggregation_method:
            raise ValueError("aggregation_method must not be empty")

        total_count = self._resolve_total_count(vehicle_counts)
        capacity = self._resolve_capacity(road_id, road_capacity)
        density = total_count / capacity
        level = self._classify(density)
        factor = self._congestion_factor(density)

        return CongestionResult(
            road_id=road_id,
            window_start=float(window_start),
            window_end=float(window_end),
            total_vehicle_count=total_count,
            road_capacity=capacity,
            traffic_density=density,
            congestion_level=level,
            congestion_factor=factor,
            aggregation_method=aggregation_method,
        )

    @staticmethod
    def dynamic_travel_time(result: CongestionResult, base_travel_time: float) -> float:
        if isinstance(base_travel_time, bool) or not isinstance(base_travel_time, (int, float)) or not isfinite(float(base_travel_time)):
            raise TypeError("base_travel_time must be a finite number")
        if base_travel_time < 0:
            raise ValueError("base_travel_time must be >= 0")
        return float(base_travel_time) * result.congestion_factor

    @staticmethod
    def _resolve_total_count(vehicle_counts: VehicleCountsInput) -> int:
        if isinstance(vehicle_counts, bool):
            raise TypeError("vehicle count cannot be bool")
        if isinstance(vehicle_counts, (int, float)):
            if not isfinite(float(vehicle_counts)):
                raise ValueError("vehicle count must be finite")
            if vehicle_counts < 0:
                raise ValueError("vehicle count cannot be negative")
            return int(round(vehicle_counts))
        if isinstance(vehicle_counts, dict):
            if "total" in vehicle_counts:
                value = vehicle_counts["total"]
            else:
                value = sum(v for k, v in vehicle_counts.items() if k in {"car", "motorcycle", "bus", "truck"})
            return CongestionEstimator._coerce_count(value)
        if hasattr(vehicle_counts, "total"):
            return CongestionEstimator._coerce_count(getattr(vehicle_counts, "total"))
        raise TypeError("vehicle_counts must be a number, dict, or object with a numeric .total")

    @staticmethod
    def _coerce_count(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("vehicle count must be numeric")
        if not isfinite(float(value)):
            raise ValueError("vehicle count must be finite")
        if value < 0:
            raise ValueError("vehicle count cannot be negative")
        return int(round(value))

    def _resolve_capacity(self, road_id: str, road_capacity: Optional[float]) -> float:
        if road_capacity is None:
            capacity = self.config.default_capacity_by_road.get(road_id, self.config.default_capacity)
        else:
            capacity = road_capacity
        if isinstance(capacity, bool) or not isinstance(capacity, (int, float)) or not isfinite(float(capacity)) or float(capacity) <= 0:
            raise ValueError("road_capacity must be a finite number > 0")
        return float(capacity)

    def _classify(self, density: float) -> CongestionLevel:
        if density <= self.config.free_flow_max:
            return CongestionLevel.FREE_FLOW
        if density <= self.config.moderate_max:
            return CongestionLevel.MODERATE
        if density <= self.config.heavy_max:
            return CongestionLevel.HEAVY
        return CongestionLevel.SEVERE

    def _congestion_factor(self, density: float) -> float:
        factor = 1.0 + self.config.alpha * (density ** self.config.beta)
        return min(max(1.0, factor), self.config.max_congestion_factor)
