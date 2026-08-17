"""Transparent junction risk scoring for NovaRoute.AI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Any, Dict, Mapping, Optional


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskConfig:
    weight_accident_history: float = 0.35
    weight_traffic_density: float = 0.25
    weight_pedestrian_conflict: float = 0.25
    weight_time_of_day: float = 0.15
    max_accidents_for_max_risk: float = 10.0
    max_density_for_max_risk: float = 1.5
    max_pedestrian_for_max_risk: float = 50.0
    time_of_day_risk_profile: tuple = (
        0.20, 0.15, 0.10, 0.10, 0.15, 0.30,
        0.55, 0.75, 0.85, 0.60, 0.45, 0.45,
        0.50, 0.45, 0.45, 0.55, 0.70, 0.85,
        0.80, 0.60, 0.45, 0.40, 0.35, 0.25,
    )
    low_max: float = 30.0
    moderate_max: float = 55.0
    high_max: float = 75.0

    def __post_init__(self) -> None:
        weights = [
            self.weight_accident_history,
            self.weight_traffic_density,
            self.weight_pedestrian_conflict,
            self.weight_time_of_day,
        ]
        if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not isfinite(float(w)) or w < 0 for w in weights):
            raise ValueError("risk weights must be finite numbers >= 0")
        if sum(weights) <= 0:
            raise ValueError("at least one risk weight must be > 0")
        for name, value in {
            "max_accidents_for_max_risk": self.max_accidents_for_max_risk,
            "max_density_for_max_risk": self.max_density_for_max_risk,
            "max_pedestrian_for_max_risk": self.max_pedestrian_for_max_risk,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be a finite number > 0")
        if len(self.time_of_day_risk_profile) != 24 or any(not 0 <= float(x) <= 1 for x in self.time_of_day_risk_profile):
            raise ValueError("time_of_day_risk_profile must contain 24 values in [0, 1]")
        if not (0 <= self.low_max <= self.moderate_max <= self.high_max <= 100):
            raise ValueError("risk thresholds must satisfy 0 <= low <= moderate <= high <= 100")


@dataclass
class RiskFactorScores:
    accident_history: float
    traffic_density: float
    pedestrian_conflict: float
    time_of_day: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RiskContributions:
    accident_history: float
    traffic_density: float
    pedestrian_conflict: float
    time_of_day: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RiskResult:
    junction_id: str
    overall_risk_score: float
    risk_level: RiskLevel
    factor_scores: RiskFactorScores
    contributions: RiskContributions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "junction_id": self.junction_id,
            "risk_score": self.overall_risk_score,
            "risk_level": self.risk_level.value,
            "factor_scores": self.factor_scores.to_dict(),
            "contributions": self.contributions.to_dict(),
        }

    def explain(self) -> str:
        c = self.contributions
        return (
            f"Final risk score: {self.overall_risk_score:.2f}/100 ({self.risk_level.value}). "
            f"Accident history contributed {c.accident_history:.2f}, "
            f"traffic density {c.traffic_density:.2f}, "
            f"pedestrian conflict {c.pedestrian_conflict:.2f}, "
            f"time of day {c.time_of_day:.2f}."
        )


class RiskScorer:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        raw = {
            "accident_history": self.config.weight_accident_history,
            "traffic_density": self.config.weight_traffic_density,
            "pedestrian_conflict": self.config.weight_pedestrian_conflict,
            "time_of_day": self.config.weight_time_of_day,
        }
        total = sum(raw.values())
        self._weights = {key: value / total for key, value in raw.items()}

    def score(
        self,
        junction_id: str,
        accident_count: float,
        traffic_density: Any,
        pedestrian_count: float,
        hour: Any,
    ) -> RiskResult:
        if not isinstance(junction_id, str) or not junction_id.strip():
            raise ValueError("junction_id must be a non-empty string")
        accident = self._non_negative_number(accident_count, "accident_count")
        pedestrian = self._non_negative_number(pedestrian_count, "pedestrian_count")
        density = self._resolve_density(traffic_density)
        hour_int = self._resolve_hour(hour)

        factors = RiskFactorScores(
            accident_history=self._normalize(accident, self.config.max_accidents_for_max_risk),
            traffic_density=self._normalize(density, self.config.max_density_for_max_risk),
            pedestrian_conflict=self._normalize(pedestrian, self.config.max_pedestrian_for_max_risk),
            time_of_day=float(self.config.time_of_day_risk_profile[hour_int]),
        )
        contributions = RiskContributions(
            accident_history=factors.accident_history * self._weights["accident_history"] * 100,
            traffic_density=factors.traffic_density * self._weights["traffic_density"] * 100,
            pedestrian_conflict=factors.pedestrian_conflict * self._weights["pedestrian_conflict"] * 100,
            time_of_day=factors.time_of_day * self._weights["time_of_day"] * 100,
        )
        score_value = min(100.0, max(0.0, sum(asdict(contributions).values())))
        return RiskResult(
            junction_id=junction_id,
            overall_risk_score=score_value,
            risk_level=self._classify(score_value),
            factor_scores=factors,
            contributions=contributions,
        )

    def _classify(self, score: float) -> RiskLevel:
        if score <= self.config.low_max:
            return RiskLevel.LOW
        if score <= self.config.moderate_max:
            return RiskLevel.MODERATE
        if score <= self.config.high_max:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    @staticmethod
    def _normalize(value: float, cap: float) -> float:
        return min(1.0, max(0.0, value / cap))

    @staticmethod
    def _non_negative_number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
        return float(value)

    @staticmethod
    def _resolve_density(value: Any) -> float:
        if isinstance(value, Mapping):
            if "traffic_density" not in value:
                raise ValueError("traffic density mapping must contain 'traffic_density'")
            value = value["traffic_density"]
        elif hasattr(value, "traffic_density"):
            value = value.traffic_density
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("traffic_density must be numeric, a mapping, or an object with traffic_density")
        if not isfinite(float(value)):
            raise ValueError("traffic_density must be finite")
        if value < 0:
            raise ValueError("traffic_density cannot be negative")
        return float(value)

    @staticmethod
    def _resolve_hour(hour: Any) -> int:
        if isinstance(hour, datetime):
            return hour.hour
        if isinstance(hour, bool) or not isinstance(hour, int):
            raise TypeError("hour must be an integer 0-23 or datetime")
        if not 0 <= hour <= 23:
            raise ValueError("hour must be between 0 and 23")
        return hour


def build_risk_config(cfg: dict) -> RiskConfig:
    section = cfg.get("risk", {}) or {}
    weights = section.get("weights", {}) or {}
    normalization = section.get("normalization", {}) or {}
    thresholds = section.get("risk_level_thresholds", {}) or {}
    profile = tuple(section.get("time_of_day_risk_profile", RiskConfig.time_of_day_risk_profile))
    return RiskConfig(
        weight_accident_history=weights.get("accident_history", RiskConfig.weight_accident_history),
        weight_traffic_density=weights.get("traffic_density", RiskConfig.weight_traffic_density),
        weight_pedestrian_conflict=weights.get("pedestrian_conflict", RiskConfig.weight_pedestrian_conflict),
        weight_time_of_day=weights.get("time_of_day", RiskConfig.weight_time_of_day),
        max_accidents_for_max_risk=normalization.get("max_accidents_for_max_risk", RiskConfig.max_accidents_for_max_risk),
        max_density_for_max_risk=normalization.get("max_density_for_max_risk", RiskConfig.max_density_for_max_risk),
        max_pedestrian_for_max_risk=normalization.get("max_pedestrian_for_max_risk", RiskConfig.max_pedestrian_for_max_risk),
        time_of_day_risk_profile=profile,
        low_max=thresholds.get("low_max", RiskConfig.low_max),
        moderate_max=thresholds.get("moderate_max", RiskConfig.moderate_max),
        high_max=thresholds.get("high_max", RiskConfig.high_max),
    )
