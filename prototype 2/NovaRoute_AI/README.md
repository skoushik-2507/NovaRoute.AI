# NovaRoute.AI

**AI-Based Traffic Risk Heatmap & Police Deployment Decision Support — Nagpur City**

Smart India Hackathon technical implementation.

## Architecture

```text
Traffic video
    ↓
YOLO vehicle detection
    ↓
ByteTrack vehicle tracking
    ↓
Average active vehicle measurement
    ↓
BPR congestion factor
    ↓
Transparent junction risk score
    ↓
Shared JSON contract
    ↓
OSMnx road graph + Dijkstra
    ↓
Officer allocation / redeployment
    ↓
React + Leaflet control-room dashboard
```

The ML module deliberately supplies explainable traffic observations rather than replacing the road-network/Dijkstra layer.

## Repository layout

- `ml/` — YOLO, ByteTrack, vehicle measurement, congestion and risk scoring.
- `integration/schemas/` — shared JSON contract for the road-network/backend/dashboard team.
- `tests/` — deterministic unit tests for the ML logic.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

Then run the ML pipeline as documented in `ml/README.md`.

## Hackathon caveat

This is a decision-support prototype. The accident-history and pedestrian-conflict inputs, road capacities, time-of-day profile and model performance must be calibrated/validated using appropriate Nagpur data before operational deployment.
