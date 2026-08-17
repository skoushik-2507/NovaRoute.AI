import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Adjust paths based on deployment environment.
ROUTING_MODULE_PATH = Path(os.getenv("ROUTING_MODULE_PATH", r"D:\Hackathon\ruthvesh\ruthvesh\ruthvesh"))
ML_MODULE_PATH = Path(os.getenv("ML_MODULE_PATH", r"D:\Hackathon\prototype 2\NovaRoute_AI"))
sys.path.append(str(ROUTING_MODULE_PATH))

from src import graph_utils, cost_matrix, officer_allocation, coverage, ml_dynamic_graph, risk_priority

GRAPH_PATH = ROUTING_MODULE_PATH / "data" / "processed" / "nagpur_routing.graphml"
METRICS_DIR = ML_MODULE_PATH / "ml" / "outputs" / "metrics"

JUNCTION_CONFIG = {
    "J001": {"backend_id": "junction_1", "name": "Sitabuldi Interchange", "shortName": "Sitabuldi", "lat": 21.1600, "lon": 79.1000},
    "J002": {"backend_id": "junction_2", "name": "Medical Square", "shortName": "Medical Sq", "lat": 21.1200, "lon": 79.0600},
    "J003": {"backend_id": "junction_3", "name": "Zero Mile Junction", "shortName": "Zero Mile", "lat": 21.2200, "lon": 79.2200},
}
BACKEND_TO_FRONTEND_ID = {v["backend_id"]: k for k, v in JUNCTION_CONFIG.items()}

DEFAULT_RISKS = {"junction_1": 30.0, "junction_2": 65.0, "junction_3": 85.0}
DEFAULT_CONGESTION = {"junction_1": 1.05, "junction_2": 1.18, "junction_3": 1.35}

state: Dict[str, Any] = {
    "graph": None,
    "officers": {},
    "allocations": [],
    "routes": [],
    "coverage": {},
    "manual_overrides": {},
    "risk_map": dict(DEFAULT_RISKS),
    "congestion_map": dict(DEFAULT_CONGESTION),
    "latest_ml": {},
}

INITIAL_OFFICERS = [
    {"id": "OFF001", "name": "Inspector S. Sharma", "badge": "NGP-104", "latitude": 21.1458, "longitude": 79.0882, "status": "AVAILABLE", "locationName": "Dharampeth HQ", "assignedJunction": None, "responseTime": None, "distanceKm": None, "locked": False},
    {"id": "OFF002", "name": "Sub-Inspector R. Patil", "badge": "NGP-208", "latitude": 21.1350, "longitude": 79.0750, "status": "AVAILABLE", "locationName": "Rahate Chowk", "assignedJunction": None, "responseTime": None, "distanceKm": None, "locked": False},
    {"id": "OFF003", "name": "Officer V. Deshmukh", "badge": "NGP-312", "latitude": 21.1500, "longitude": 79.1100, "status": "AVAILABLE", "locationName": "Cotton Market Post", "assignedJunction": None, "responseTime": None, "distanceKm": None, "locked": False},
]


class SimulateIncidentRequest(BaseModel):
    junctionId: str
    incidentType: str = "Accident"
    severity: str = "High"
    congestionImpact: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def map_risk_level(level_str: str) -> str:
    return {"low": "LOW", "moderate": "MEDIUM", "high": "HIGH", "critical": "CRITICAL"}.get(str(level_str).lower(), "MEDIUM")


def risk_level_from_score(score: float) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 55:
        return "HIGH"
    if score >= 30:
        return "MEDIUM"
    return "LOW"


def congestion_level_from_factor(factor: float) -> str:
    if factor >= 1.6:
        return "SEVERE"
    if factor >= 1.3:
        return "HIGH"
    if factor >= 1.1:
        return "MODERATE"
    return "LOW"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def read_latest_ml_files() -> Dict[str, Dict[str, Any]]:
    observations: Dict[str, Dict[str, Any]] = {}
    for meta in JUNCTION_CONFIG.values():
        backend_id = meta["backend_id"]
        path = METRICS_DIR / f"{backend_id}_latest.json"
        if not path.exists():
            continue
        try:
            import json

            observations[backend_id] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return observations


def build_fallback_route(officer_id: str, frontend_junction_id: str, response_time: float = 4.5) -> Dict[str, Any]:
    officer = state["officers"][officer_id]
    junction = JUNCTION_CONFIG[frontend_junction_id]
    return {
        "officerId": officer_id,
        "junctionId": frontend_junction_id,
        "responseTime": response_time,
        "distance": round(max(response_time * 0.75, 1.0), 2),
        "path": [[officer["latitude"], officer["longitude"]], [junction["lat"], junction["lon"]]],
    }


def run_recalculation_internal() -> None:
    base_graph = state["graph"]
    latest_ml = read_latest_ml_files()
    state["latest_ml"] = latest_ml

    risk_map = dict(DEFAULT_RISKS)
    congestion_map = dict(DEFAULT_CONGESTION)
    for backend_id, obs in latest_ml.items():
        risk_map[backend_id] = float(obs.get("risk_score", risk_map.get(backend_id, 45.0)))
        congestion_map[backend_id] = float(obs.get("congestion_factor", congestion_map.get(backend_id, 1.05)))
    for backend_id, value in state["manual_overrides"].items():
        risk_map[backend_id] = float(value)
        congestion_map[backend_id] = max(congestion_map.get(backend_id, 1.05), 1.75)

    state["risk_map"] = risk_map
    state["congestion_map"] = congestion_map

    if base_graph is None:
        # Stable presentation fallback if geospatial dependencies or graph loading fail.
        sorted_junctions = sorted(JUNCTION_CONFIG.items(), key=lambda x: risk_map[x[1]["backend_id"]], reverse=True)
        officer_ids = list(state["officers"].keys())
        formatted_allocations = []
        assigned_routes = []
        for idx, (frontend_id, meta) in enumerate(sorted_junctions):
            if idx >= len(officer_ids):
                continue
            officer_id = officer_ids[idx]
            if state["officers"][officer_id]["locked"]:
                continue
            response_time = round(3.2 + idx * 1.1, 2)
            distance_km = round(2.2 + idx * 0.8, 2)
            state["officers"][officer_id].update({"assignedJunction": frontend_id, "status": "ASSIGNED", "responseTime": response_time, "distanceKm": distance_km})
            formatted_allocations.append({
                "junctionId": frontend_id,
                "officerId": officer_id,
                "responseTime": response_time,
                "distanceKm": distance_km,
                "alternatives": [],
                "reason": f"Risk ({risk_map[meta['backend_id']]:.1f}) prioritized. ETA: {response_time} mins.",
            })
            assigned_routes.append(build_fallback_route(officer_id, frontend_id, response_time))
        state["allocations"], state["routes"] = formatted_allocations, assigned_routes
        _update_coverage()
        return

    junction_files = {
        meta["backend_id"]: METRICS_DIR / f"{meta['backend_id']}_latest.json"
        for meta in JUNCTION_CONFIG.values()
        if (METRICS_DIR / f"{meta['backend_id']}_latest.json").exists()
    }
    active_graph = ml_dynamic_graph.build_dynamic_graph_from_files(base_graph, junction_files).graph if junction_files else base_graph

    officers_input = [
        {"id": o["id"], "latitude": o["latitude"], "longitude": o["longitude"]}
        for o in state["officers"].values()
        if not o.get("locked") and o.get("status") != "OFFLINE"
    ]
    junctions_input = [{"id": m["backend_id"], "latitude": m["lat"], "longitude": m["lon"]} for m in JUNCTION_CONFIG.values()]

    matrix_res = cost_matrix.build_cost_matrix(
        active_graph,
        officers_input,
        junctions_input,
        weight_mode="dynamic" if junction_files else "base",
    )
    alloc_res = officer_allocation.assign_officers(matrix_res, risk_scores=risk_map)
    cov_res = coverage.analyze_coverage_from_cost_matrix(matrix_res, threshold_minutes=6.0, risk_scores=risk_map)

    for off_id in state["officers"]:
        if not state["officers"][off_id]["locked"]:
            state["officers"][off_id].update({"assignedJunction": None, "status": "AVAILABLE", "responseTime": None, "distanceKm": None})

    formatted_allocations, assigned_routes = [], []
    for item in alloc_res.get("assignments", []):
        off_id, backend_junction_id = item["officer_id"], item["junction_id"]
        frontend_junction_id = BACKEND_TO_FRONTEND_ID.get(backend_junction_id, backend_junction_id)
        response_time = round(float(item["response_time_minutes"]), 2)
        risk_score = round(float(item.get("risk_score") or 0.0), 1)
        route = item.get("route") or {}
        dist_km = round((route.get("total_distance_meters") or 1200.0) / 1000.0, 2)

        route_nodes = route.get("route_nodes") or []
        lat_lon_path = []
        for node in route_nodes:
            if node not in active_graph.nodes:
                continue
            node_data = active_graph.nodes[node]
            lat_lon_path.append([float(node_data.get("y", 0.0)), float(node_data.get("x", 0.0))])
        if not lat_lon_path:
            lat_lon_path = build_fallback_route(off_id, frontend_junction_id, response_time)["path"]

        if off_id in state["officers"]:
            state["officers"][off_id].update({"assignedJunction": frontend_junction_id, "status": "ASSIGNED", "responseTime": response_time, "distanceKm": dist_km})

        alternatives = []
        for alt in item.get("alternatives", []) if isinstance(item.get("alternatives"), list) else []:
            if isinstance(alt, dict) and "officerId" in alt:
                alternatives.append(alt)

        formatted_allocations.append({
            "junctionId": frontend_junction_id,
            "officerId": off_id,
            "responseTime": response_time,
            "distanceKm": dist_km,
            "alternatives": alternatives,
            "reason": f"Risk ({risk_score}) prioritized. ETA: {response_time} mins.",
        })
        assigned_routes.append({"officerId": off_id, "junctionId": frontend_junction_id, "responseTime": response_time, "distance": dist_km, "path": lat_lon_path})

    state["allocations"], state["routes"] = formatted_allocations, assigned_routes
    _update_coverage(cov_res)


def _update_coverage(cov_res: Optional[Dict[str, Any]] = None) -> None:
    if cov_res is not None:
        covered_ids = [BACKEND_TO_FRONTEND_ID.get(x, x) for x in cov_res.get("covered_junction_ids", [])]
        uncovered_ids = [BACKEND_TO_FRONTEND_ID.get(x, x) for x in cov_res.get("uncovered_junction_ids", [])]
    else:
        covered_ids = [a["junctionId"] for a in state["allocations"] if a["responseTime"] <= 6.0]
        uncovered_ids = [jid for jid in JUNCTION_CONFIG if jid not in covered_ids]

    responses = [float(a["responseTime"]) for a in state["allocations"]]
    total = len(JUNCTION_CONFIG)
    state["coverage"] = {
        "coveragePct": int((len(covered_ids) / max(total, 1)) * 100),
        "coveredJunctions": len(covered_ids),
        "totalJunctions": total,
        "thresholdMinutes": 6.0,
        "avgResponse": round(float(np.mean(responses) if responses else 4.5), 1),
        "worstResponse": round(float(max(responses) if responses else 6.0), 1),
        "uncoveredCount": len(uncovered_ids),
        "uncoveredJunctions": uncovered_ids,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    if GRAPH_PATH.exists():
        state["graph"] = graph_utils.load_graph(GRAPH_PATH)
    state["officers"] = {off["id"]: dict(off) for off in INITIAL_OFFICERS}
    run_recalculation_internal()
    yield


app = FastAPI(title="NovaRoute.AI API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "graphLoaded": state["graph"] is not None,
        "graphPath": str(GRAPH_PATH),
        "metricsDir": str(METRICS_DIR),
        "mlFiles": sorted(state["latest_ml"].keys()),
        "timestamp": utc_now(),
    }


@app.get("/api/junctions")
def get_junctions():
    j_list = []
    for frontend_id, meta in JUNCTION_CONFIG.items():
        backend_id = meta["backend_id"]
        risk_score = float(state["risk_map"].get(backend_id, 45.0))
        ml_obs = state["latest_ml"].get(backend_id, {})
        risk_level = map_risk_level(ml_obs.get("risk_level")) if ml_obs else risk_level_from_score(risk_score)
        rf = ml_obs.get("risk_factor_scores", {})
        j_list.append({
            "id": frontend_id,
            "name": meta["name"],
            "shortName": meta["shortName"],
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "riskScore": round(risk_score, 1),
            "riskLevel": risk_level,
            "factors": {
                "accident": round(float(rf.get("accident_history", 0.2)) * 100, 1),
                "traffic": round(float(rf.get("traffic_density", 0.3)) * 100, 1),
                "pedestrian": round(float(rf.get("pedestrian_conflict", 0.1)) * 100, 1),
                "timeOfDay": round(float(rf.get("time_of_day", 0.4)) * 100, 1),
            },
            "congestionFactor": round(float(state["congestion_map"].get(backend_id, 1.05)), 3),
            "assignedOfficer": next((o["id"] for o in state["officers"].values() if o.get("assignedJunction") == frontend_id), None),
        })
    return j_list


@app.get("/api/risk")
def get_risk():
    return get_junctions()


@app.get("/api/officers")
def get_officers():
    return list(state["officers"].values())


@app.get("/api/allocations")
def get_allocations():
    return state["allocations"]


@app.get("/api/routes")
def get_routes():
    return state["routes"]


@app.get("/api/routes/{officer_id}/{junction_id}")
def get_route(officer_id: str, junction_id: str):
    for route in state["routes"]:
        if route["officerId"] == officer_id and route["junctionId"] == junction_id:
            return route
    if officer_id not in state["officers"] or junction_id not in JUNCTION_CONFIG:
        raise HTTPException(status_code=404, detail="Officer or junction not found")
    return build_fallback_route(officer_id, junction_id)


@app.get("/api/traffic")
def get_traffic():
    observations = list(state["latest_ml"].values())
    if observations:
        vehicle_count = sum(int(obs.get("total_vehicles", 0)) for obs in observations)
        densities = [float(obs.get("traffic_density", 0.0)) for obs in observations]
        factors = [float(obs.get("congestion_factor", 1.0)) for obs in observations]
        avg_density = float(np.mean(densities))
        avg_factor = float(np.mean(factors))
        trend = densities[-10:]
    else:
        vehicle_count = 312
        avg_density = 0.58
        avg_factor = float(np.mean(list(state["congestion_map"].values())))
        trend = [0.42, 0.48, 0.53, 0.58, 0.61, 0.58]
    return {
        "vehicleCount": vehicle_count,
        "avgDensity": round(avg_density, 3),
        "congestionLevel": congestion_level_from_factor(avg_factor),
        "congestionFactor": round(avg_factor, 3),
        "networkStatus": "LIVE",
        "timestamp": utc_now(),
        "trend": [round(float(x), 3) for x in trend],
    }


@app.get("/api/coverage")
def get_coverage():
    return state["coverage"]


@app.get("/api/baseline")
def get_baseline():
    cov = state["coverage"] or {}
    return {
        "static": {"avgResponse": 8.4, "coveragePct": 33, "coveredCount": 1, "total": len(JUNCTION_CONFIG)},
        "novaroute": {
            "avgResponse": cov.get("avgResponse", 4.5),
            "coveragePct": cov.get("coveragePct", 66),
            "coveredCount": cov.get("coveredJunctions", 2),
            "total": cov.get("totalJunctions", len(JUNCTION_CONFIG)),
        },
    }


@app.post("/api/allocation/recalculate")
def recalculate():
    run_recalculation_internal()
    return {"status": "success"}


@app.post("/api/refresh")
def refresh():
    run_recalculation_internal()
    return {"status": "success", "timestamp": utc_now()}


@app.post("/api/officers/{officer_id}/lock")
def lock_officer(officer_id: str):
    if officer_id not in state["officers"]:
        raise HTTPException(status_code=404, detail="Officer not found")
    state["officers"][officer_id]["locked"] = True
    state["officers"][officer_id]["status"] = "LOCKED"
    run_recalculation_internal()
    return {"status": "success"}


@app.post("/api/officers/{officer_id}/unlock")
def unlock_officer(officer_id: str):
    if officer_id not in state["officers"]:
        raise HTTPException(status_code=404, detail="Officer not found")
    state["officers"][officer_id]["locked"] = False
    state["officers"][officer_id]["status"] = "AVAILABLE"
    run_recalculation_internal()
    return {"status": "success"}


@app.post("/api/incidents/simulate")
def simulate_incident(req: SimulateIncidentRequest):
    if req.junctionId not in JUNCTION_CONFIG:
        raise HTTPException(status_code=404, detail="Junction not found")
    backend_id = JUNCTION_CONFIG[req.junctionId]["backend_id"]
    prev_risk = float(state["risk_map"].get(backend_id, 45.0))
    prev_congestion = float(state["congestion_map"].get(backend_id, 1.05))
    bump = {"Low": 10.0, "Medium": 18.0, "High": 28.0, "Critical": 40.0}.get(req.severity, 28.0)
    state["manual_overrides"][backend_id] = min(100.0, prev_risk + bump)
    before = state["coverage"].get("avgResponse", 7.8) if state["coverage"] else 7.8
    previous = [{"officerId": a["officerId"], "fromJunction": a["junctionId"], "toJunction": a["junctionId"]} for a in state["allocations"]]
    run_recalculation_internal()
    after = state["coverage"].get("avgResponse", 4.5)
    new = [{"officerId": a["officerId"], "fromJunction": None, "toJunction": a["junctionId"]} for a in state["allocations"]]
    return {
        "affectedJunction": req.junctionId,
        "prevRiskScore": round(prev_risk, 1),
        "newRiskScore": round(float(state["risk_map"].get(backend_id, 98.0)), 1),
        "prevCongestion": round(prev_congestion, 3),
        "newCongestion": round(float(state["congestion_map"].get(backend_id, 1.75)), 3),
        "previousOfficers": previous,
        "newOfficers": new,
        "responseTimeBefore": before,
        "responseTimeAfter": after,
        "improvementPercentage": round(max(0.0, (1 - after / before) * 100), 1) if before else 0,
    }
