from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, MongoClient

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RUTHVESH_ROOT = Path(os.getenv("RUTHVESH_ROOT", ROOT / "ruthvesh"))
if str(RUTHVESH_ROOT) not in sys.path:
    sys.path.insert(0, str(RUTHVESH_ROOT))

try:
    import networkx as nx
    from src.routing import shortest_path
    from src.congestion import DYNAMIC_WEIGHT_ATTRIBUTE, calculate_dynamic_travel_time, get_segment_id
except Exception as exc:  # pragma: no cover - gives a useful startup error
    nx = None
    shortest_path = None
    DYNAMIC_WEIGHT_ATTRIBUTE = "dynamic_travel_time"
    calculate_dynamic_travel_time = None
    get_segment_id = None
    RUTHVESH_IMPORT_ERROR = str(exc)
else:
    RUTHVESH_IMPORT_ERROR = None

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "novaroute")
GRAPH_PATH = Path(os.getenv(
    "RUTHVESH_GRAPH",
    RUTHVESH_ROOT / "data" / "processed" / "nagpur_routing.graphml",
))

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2500)
db = client[MONGODB_DB]
junctions_col = db["junctions"]
officers_col = db["officers"]
traffic_col = db["traffic_observations"]
incidents_col = db["incidents"]

app = FastAPI(title="NovaRoute.AI Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_seed() -> dict:
    return json.loads((ROOT / "seed_data.json").read_text(encoding="utf-8"))


def ensure_seed_data() -> None:
    """Seed MongoDB once so the final UI works immediately after setup."""
    seed = load_seed()
    if junctions_col.count_documents({}) == 0:
        junctions_col.insert_many(seed["junctions"])
    if officers_col.count_documents({}) == 0:
        officers_col.insert_many(seed["officers"])
    if traffic_col.count_documents({}) == 0:
        traffic = dict(seed["traffic"])
        traffic.update({
            "road_segment_id": "network",
            "schema_version": "seed",
            "storedAt": utc_now(),
        })
        traffic_col.insert_one(traffic)
    junctions_col.create_index([("id", ASCENDING)], unique=True)
    officers_col.create_index([("id", ASCENDING)], unique=True)
    traffic_col.create_index([("road_segment_id", ASCENDING), ("timestamp", DESCENDING)])
    incidents_col.create_index([("timestamp", DESCENDING)])


def mongo_health() -> bool:
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False


def normalize_junction_id(value: str) -> str:
    v = str(value)
    if v.startswith("junction_") and v[len("junction_"):].isdigit():
        return f"J{int(v[len('junction_'):]):03d}"
    return v


def risk_level(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def congestion_level(factor: float) -> str:
    if factor >= 1.6:
        return "SEVERE"
    if factor >= 1.35:
        return "HIGH"
    if factor >= 1.15:
        return "MODERATE"
    return "LOW"


def latest_observations() -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    for doc in traffic_col.find({}).sort("timestamp", DESCENDING):
        key = normalize_junction_id(doc.get("road_segment_id", ""))
        if key in ("network", "") or key in latest:
            continue
        latest[key] = doc
    return latest


def enrich_junction(j: dict, obs: Optional[dict]) -> dict:
    out = {k: v for k, v in j.items() if k != "_id"}
    if obs:
        out["riskScore"] = float(obs.get("risk_score", out.get("riskScore", 0)))
        out["riskLevel"] = risk_level(out["riskScore"])
        rf = obs.get("risk_factor_scores", {})
        out["factors"] = {
            "accident": round(float(rf.get("accident_history", 0)) * 100, 2),
            "traffic": round(float(rf.get("traffic_density", 0)) * 100, 2),
            "pedestrian": round(float(rf.get("pedestrian_conflict", 0)) * 100, 2),
            "timeOfDay": round(float(rf.get("time_of_day", 0)) * 100, 2),
        }
        out["congestionFactor"] = float(obs.get("congestion_factor", out.get("congestionFactor", 1.0)))
    return out


def load_graph():
    if nx is None:
        raise RuntimeError(f"Ruthvesh routing dependencies are unavailable: {RUTHVESH_IMPORT_ERROR}")
    if not GRAPH_PATH.exists():
        raise RuntimeError(f"Processed Ruthvesh graph not found: {GRAPH_PATH}")
    return nx.read_graphml(GRAPH_PATH)


def nearest_node(graph, lat: float, lon: float):
    best = None
    best_d = float("inf")
    for node, data in graph.nodes(data=True):
        try:
            x = float(data.get("x"))
            y = float(data.get("y"))
        except (TypeError, ValueError):
            continue
        d = (y - lat) ** 2 + (x - lon) ** 2
        if d < best_d:
            best_d = d
            best = node
    return best


def build_dynamic_graph(graph):
    """Use ML congestion factors stored in MongoDB on the existing Ruthvesh graph."""
    dynamic = graph.copy()
    for u, v, key, data in dynamic.edges(keys=True, data=True):
        if data.get("travel_time") is not None:
            data[DYNAMIC_WEIGHT_ATTRIBUTE] = float(data["travel_time"])

    # Prefer explicit OSM edge ids supplied by the ML contract.
    for doc in traffic_col.find({"osm_edge": {"$ne": None}}).sort("timestamp", DESCENDING):
        edge = doc.get("osm_edge") or {}
        if "u" not in edge or "v" not in edge:
            continue
        u, v = str(edge["u"]), str(edge["v"])
        key = edge.get("key")
        factor = float(doc.get("congestion_factor", 1.0))
        if not dynamic.has_edge(u, v):
            # GraphML may deserialize numeric OSM ids as strings; try int ids.
            try:
                u2, v2 = int(edge["u"]), int(edge["v"])
                if dynamic.has_edge(u2, v2):
                    u, v = u2, v2
            except Exception:
                pass
        if not dynamic.has_edge(u, v):
            continue
        candidates = dynamic.get_edge_data(u, v)
        keys = [key] if key in candidates else list(candidates.keys()) if key is None else []
        for k in keys:
            base = candidates[k].get("travel_time")
            if base is not None:
                candidates[k][DYNAMIC_WEIGHT_ATTRIBUTE] = float(base) * factor

    return dynamic


def route_pair(graph, officer: dict, junction: dict, dynamic: bool = True, routing_graph=None) -> Optional[dict]:
    origin = nearest_node(graph, float(officer["latitude"]), float(officer["longitude"]))
    dest = nearest_node(graph, float(junction["latitude"]), float(junction["longitude"]))
    if origin is None or dest is None:
        return None
    g = routing_graph if routing_graph is not None else (build_dynamic_graph(graph) if dynamic else graph)
    result = shortest_path(g, origin, dest, weight_mode="dynamic" if dynamic else "base")
    if not result.get("is_reachable"):
        return None
    path = []
    for node in result["route_nodes"]:
        d = g.nodes[node]
        try:
            path.append([float(d["y"]), float(d["x"])])
        except (TypeError, ValueError):
            continue
    return {
        "officerId": officer["id"],
        "junctionId": junction["id"],
        "responseTime": round(float(result["total_time_seconds"]) / 60.0, 2),
        "distance": round(float(result["total_distance_meters"]) / 1000.0, 3),
        "path": path,
    }


@app.on_event("startup")
def startup() -> None:
    if mongo_health():
        ensure_seed_data()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "mongodb": mongo_health(),
        "ruthveshGraph": GRAPH_PATH.exists(),
        "ruthveshImport": RUTHVESH_IMPORT_ERROR is None,
        "timestamp": utc_now(),
    }


@app.get("/api/junctions")
def get_junctions() -> list:
    obs = latest_observations()
    return [enrich_junction(j, obs.get(j["id"])) for j in junctions_col.find({}).sort("id", ASCENDING)]


@app.get("/api/risk")
def get_risk() -> list:
    return get_junctions()


@app.get("/api/officers")
def get_officers() -> list:
    return [{k: v for k, v in o.items() if k != "_id"} for o in officers_col.find({}).sort("id", ASCENDING)]


@app.get("/api/traffic")
def get_traffic() -> dict:
    network = traffic_col.find_one({"road_segment_id": "network"}, sort=[("timestamp", DESCENDING)])
    docs = list(traffic_col.find({"road_segment_id": {"$ne": "network"}}).sort("timestamp", DESCENDING).limit(100))
    if not network:
        network = {}
    vehicle_count = sum(int(d.get("total_vehicles", 0)) for d in docs[-20:]) if docs else int(network.get("vehicleCount", 0))
    densities = [float(d.get("traffic_density", 0)) for d in docs]
    factors = [float(d.get("congestion_factor", 1.0)) for d in docs]
    avg_density = sum(densities) / len(densities) if densities else float(network.get("avgDensity", 0))
    avg_factor = sum(factors) / len(factors) if factors else float(network.get("congestionFactor", 1.0))
    return {
        "vehicleCount": vehicle_count,
        "avgDensity": round(avg_density, 3),
        "congestionLevel": congestion_level(avg_factor),
        "congestionFactor": round(avg_factor, 3),
        "networkStatus": "LIVE" if mongo_health() else "OFFLINE",
        "timestamp": utc_now(),
        "trend": [round(float(x), 3) for x in densities[-10:]] or network.get("trend", []),
    }


@app.get("/api/traffic/{junction_id}")
def get_junction_traffic(junction_id: str) -> dict:
    jid = normalize_junction_id(junction_id)
    doc = traffic_col.find_one({"road_segment_id": {"$in": [jid, junction_id, f"junction_{int(jid[1:])}" if jid.startswith('J') and jid[1:].isdigit() else jid]}}, sort=[("timestamp", DESCENDING)])
    if not doc:
        raise HTTPException(404, "No traffic observation for junction")
    doc.pop("_id", None)
    return doc


class TrafficObservation(BaseModel):
    schema_version: str = "1.1.0"
    road_segment_id: str
    osm_edge: Optional[dict] = None
    timestamp: str
    observation_window_seconds: float = Field(gt=0)
    vehicle_counts: dict
    total_vehicles: int = Field(ge=0)
    peak_vehicles: int = Field(ge=0)
    road_capacity: float = Field(gt=0)
    traffic_density: float = Field(ge=0)
    congestion_level: str
    congestion_factor: float = Field(ge=1)
    aggregation_method: str = "average_active_vehicles"
    risk_score: float = Field(ge=0, le=100)
    risk_level: str
    risk_factor_scores: dict
    risk_contributions: dict


@app.post("/api/traffic")
def ingest_traffic(observation: TrafficObservation) -> dict:
    doc = observation.model_dump()
    doc["storedAt"] = utc_now()
    traffic_col.insert_one(doc)
    jid = normalize_junction_id(observation.road_segment_id)
    junctions_col.update_one(
        {"id": jid},
        {"$set": {
            "riskScore": observation.risk_score,
            "riskLevel": risk_level(observation.risk_score),
            "congestionFactor": observation.congestion_factor,
        }},
    )
    return {"stored": True, "junctionId": jid, "id": str(doc.get("_id"))}


@app.post("/api/traffic/bulk")
def ingest_traffic_bulk(observations: List[TrafficObservation]) -> dict:
    results = [ingest_traffic(o) for o in observations]
    return {"stored": len(results)}


@app.get("/api/routes")
def get_routes() -> list:
    try:
        graph = load_graph()
    except Exception:
        seed = load_seed()
        return seed["routes"]
    junctions = get_junctions()
    officers = get_officers()
    routes = []
    dynamic_graph = build_dynamic_graph(graph)
    for officer in officers:
        if officer.get("status") == "OFFLINE":
            continue
        jid = officer.get("assignedJunction")
        if not jid:
            continue
        junction = next((j for j in junctions if j["id"] == jid), None)
        if junction:
            r = route_pair(graph, officer, junction, dynamic=True, routing_graph=dynamic_graph)
            if r:
                routes.append(r)
    return routes


@app.get("/api/routes/{officer_id}/{junction_id}")
def get_route(officer_id: str, junction_id: str) -> dict:
    officer = officers_col.find_one({"id": officer_id})
    junction = junctions_col.find_one({"id": normalize_junction_id(junction_id)})
    if not officer or not junction:
        raise HTTPException(404, "Officer or junction not found")
    try:
        route = route_pair(load_graph(), officer, enrich_junction(junction, latest_observations().get(junction["id"])), True)
    except Exception as exc:
        raise HTTPException(503, f"Routing unavailable: {exc}") from exc
    if route is None:
        raise HTTPException(404, "No route found")
    return route


@app.get("/api/allocations")
def get_allocations() -> list:
    junctions = get_junctions()
    officers = [o for o in get_officers() if o.get("status") != "OFFLINE" and not o.get("locked")]
    graph = None
    dynamic_graph = None
    try:
        graph = load_graph()
        dynamic_graph = build_dynamic_graph(graph)
    except Exception:
        pass

    allocations = []
    used = set()
    # Highest risk first, matching Ruthvesh's priority-first allocation idea.
    for junction in sorted(junctions, key=lambda x: float(x.get("riskScore", 0)), reverse=True):
        candidates = []
        for officer in officers:
            if officer["id"] in used:
                continue
            if graph is not None:
                try:
                    r = route_pair(graph, officer, junction, True, routing_graph=dynamic_graph)
                except Exception:
                    r = None
            else:
                r = None
            if r:
                candidates.append((r["responseTime"], r["distance"], officer, r))
            elif officer.get("responseTime") is not None:
                candidates.append((float(officer["responseTime"]), float(officer.get("distanceKm") or 0), officer, None))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        best = candidates[0]
        used.add(best[2]["id"])
        allocations.append({
            "junctionId": junction["id"],
            "officerId": best[2]["id"],
            "responseTime": round(best[0], 2),
            "distanceKm": round(best[1], 3),
            "alternatives": [{"officerId": c[2]["id"], "responseTime": round(c[0], 2)} for c in candidates[1:4]],
            "reason": f"{best[2]['name']} has the shortest available Ruthvesh road-network response time for this risk-prioritized junction.",
        })
    return allocations


@app.get("/api/coverage")
def get_coverage() -> dict:
    allocations = get_allocations()
    junctions = get_junctions()
    threshold = 6.0
    times = {a["junctionId"]: a["responseTime"] for a in allocations}
    covered = [j for j in junctions if times.get(j["id"], float("inf")) <= threshold]
    responses = list(times.values())
    uncovered = [j["id"] for j in junctions if j["id"] not in times or times[j["id"]] > threshold]
    return {
        "coveragePct": round(100 * len(covered) / len(junctions), 1) if junctions else 0,
        "coveredJunctions": len(covered),
        "totalJunctions": len(junctions),
        "thresholdMinutes": threshold,
        "avgResponse": round(sum(responses) / len(responses), 2) if responses else 0,
        "worstResponse": round(max(responses), 2) if responses else 0,
        "uncoveredCount": len(uncovered),
        "uncoveredJunctions": uncovered,
    }


@app.get("/api/baseline")
def get_baseline() -> dict:
    coverage = get_coverage()
    seed = load_seed()["baseline"]
    return {
        "static": seed["static"],
        "novaroute": {
            "avgResponse": coverage["avgResponse"],
            "coveragePct": coverage["coveragePct"],
            "coveredCount": coverage["coveredJunctions"],
            "total": coverage["totalJunctions"],
        },
    }


class IncidentRequest(BaseModel):
    junctionId: str
    incidentType: str
    severity: str
    congestionImpact: bool = True


@app.post("/api/incidents/simulate")
def simulate_incident(payload: IncidentRequest) -> dict:
    jid = normalize_junction_id(payload.junctionId)
    j = junctions_col.find_one({"id": jid})
    if not j:
        raise HTTPException(404, "Junction not found")
    prev_risk = float(j.get("riskScore", 0))
    prev_cong = float(j.get("congestionFactor", 1.0))
    bump = {"Low": 8, "Medium": 15, "High": 22, "Critical": 30}.get(payload.severity, 15)
    new_risk = min(100, prev_risk + bump)
    new_cong = min(2.5, prev_cong + (0.18 if payload.congestionImpact else 0.0))
    junctions_col.update_one({"id": jid}, {"$set": {"riskScore": new_risk, "riskLevel": risk_level(new_risk), "congestionFactor": new_cong}})
    incident = payload.model_dump()
    incident.update({"timestamp": utc_now(), "status": "ACTIVE"})
    incidents_col.insert_one(incident)
    return {
        "affectedJunction": jid,
        "prevRiskScore": prev_risk,
        "newRiskScore": new_risk,
        "prevCongestion": prev_cong,
        "newCongestion": new_cong,
        "previousOfficers": [],
        "newOfficers": [],
        "responseTimeBefore": 0,
        "responseTimeAfter": 0,
        "improvementPercentage": 0,
    }


@app.post("/api/officers/{officer_id}/lock")
def lock_officer(officer_id: str) -> dict:
    result = officers_col.update_one({"id": officer_id}, {"$set": {"locked": True, "status": "LOCKED"}})
    if result.matched_count == 0:
        raise HTTPException(404, "Officer not found")
    return {"ok": True}


@app.post("/api/officers/{officer_id}/unlock")
def unlock_officer(officer_id: str) -> dict:
    result = officers_col.update_one({"id": officer_id}, {"$set": {"locked": False, "status": "AVAILABLE"}})
    if result.matched_count == 0:
        raise HTTPException(404, "Officer not found")
    return {"ok": True}


@app.post("/api/allocation/recalculate")
def recalculate() -> dict:
    return {"ok": True, "allocations": get_allocations(), "timestamp": utc_now()}


@app.post("/api/refresh")
def refresh() -> dict:
    return {"ok": True, "timestamp": utc_now()}
