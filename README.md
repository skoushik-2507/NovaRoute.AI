# NovaRoute.AI

**Smart traffic routing & officer deployment system for Nagpur.**

Built for the **Manthan4Yuva Hackathon** — combines real-time ML congestion detection with OSMnx/Dijkstra-based dynamic routing to power adaptive path optimization and emergency-response resource allocation, with a human-in-the-loop dashboard for final decision-making.

---

## Problem Statement

Urban traffic control rooms and emergency-response units in cities like Nagpur largely rely on static road data and manual judgment to decide:

- Which routes are actually fastest *right now*, not on average
- Which junctions are becoming high-risk (accidents, pedestrian conflict, congestion) in real time
- How to deploy a limited number of officers/response units to cover the highest-risk points within an acceptable response time

Static maps and fixed-time-of-day routing don't capture live congestion. Response-unit deployment is often ad hoc rather than optimized against real-time risk and actual road-network travel times. The result is slower emergency response, uneven area coverage, and decisions made without a clear, explainable basis.

## Solution

NovaRoute.AI is a decision-support pipeline that closes the loop from **live camera feed → traffic understanding → optimal routing → resource allocation → human-reviewed dashboard**:

1. **Vehicle detection & tracking** (YOLOv8 + ByteTrack) processes junction camera feeds to count vehicles by class and track them across frames.
2. **Congestion & risk scoring** turns raw vehicle counts into a standardized, schema-validated traffic observation per junction — congestion level, congestion factor, and a 0–100 risk score broken down by contributing factors (accident history, traffic density, pedestrian conflict, time of day).
3. **Dynamic road-network routing** loads Nagpur's road graph (via OSMnx) and applies the live congestion factors to edge weights, so Dijkstra-based shortest-path queries reflect real, current travel times — not just free-flow distance.
4. **Officer/resource allocation** builds an officer × high-risk-junction response-time cost matrix from real routing (not straight-line distance), analyzes coverage against a response-time threshold, and assigns officers to junctions using tiered, risk-prioritized optimization (Hungarian algorithm).
5. **Dashboard** (React + Leaflet) visualizes junctions, live risk/congestion, routes, and officer coverage, giving a human operator full visibility and final override authority over every automated recommendation.

The system is intentionally a **decision-support tool, not a fully autonomous one** — every routing and allocation recommendation is surfaced to a human operator for review before action.

## Architecture

Three core engines, integrated behind a FastAPI backend, surfaced through a React dashboard:

```
                     ┌─────────────────────────┐
  Junction camera →  │   NovaRoute_AI (ML)      │
  feeds              │  YOLOv8 + ByteTrack      │
                     │  → congestion + risk JSON│
                     └────────────┬─────────────┘
                                  │  schema-validated
                                  │  traffic observations
                                  ▼
                     ┌─────────────────────────┐
                     │  Backend (FastAPI)       │
                     │  ingest · adapt · serve  │
                     └──┬────────────────────┬──┘
                        │                    │
                        ▼                    ▼
          ┌─────────────────────┐  ┌──────────────────────┐
          │ ruthvesh (routing)  │  │ Supabase              │
          │ Nagpur road graph   │  │ PostgreSQL + PostGIS  │
          │ Dijkstra routing    │  │ Auth · Realtime       │
          │ cost matrix         │  └──────────────────────┘
          │ coverage analysis   │
          │ officer allocation  │
          └─────────────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │  Dashboard (React +      │
          │  Leaflet)                │
          │  live map, risk, routes, │
          │  officer coverage        │
          └─────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Vehicle detection & tracking | YOLOv8, ByteTrack, OpenCV |
| Congestion / risk scoring | Python, custom scoring models |
| Road network & routing | OSMnx, NetworkX (Dijkstra), GeoPandas |
| Allocation optimization | SciPy (Hungarian algorithm) |
| Backend API | FastAPI |
| Database | Supabase (PostgreSQL + PostGIS, Auth, Realtime) |
| Frontend dashboard | React, Leaflet |
| Testing | Pytest |

## Repository Structure

```
prototype 2/NovaRoute_AI/   → ML pipeline: detection, tracking, congestion, risk scoring
ruthvesh/ruthvesh/          → Road graph, Dijkstra routing, cost matrix, coverage, officer allocation
backend/                    → FastAPI service integrating ML outputs + ruthvesh + Supabase
final/, final_mongodb_connected/final_work/  → Frontend dashboard iterations
```

## Team

| Member | Role | Contribution |
|---|---|---|
| **Koushik** | ML / Frontend | YOLO-based vehicle detection pipeline and risk scoring model; contributed to frontend development |
| **Ruthvesh** | Data Pipelines & Team Lead | Built the Nagpur road graph pipeline (OSMnx) and the Dijkstra-based distance/time query engine that all routing and allocation features are built on |
| **Sreekar** | Full-Stack Integration | End-to-end integration across ML, backend, and frontend; built the live dashboard (React + Leaflet) |
| **Lochan** | Backend API | Designed and built the FastAPI backend, PostGIS-backed data layer, and the officer allocation algorithm |
| **Pranay** | Frontend | Built dashboard UI with React + Leaflet, and contributed across ML, backend, and integration work as needed throughout the hackathon |

---

*Built for Manthan4Yuva Hackathon.*
