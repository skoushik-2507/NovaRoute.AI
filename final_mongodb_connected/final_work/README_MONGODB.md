# NovaRoute.AI — Final + MongoDB + Ruthvesh

This version of the final frontend is connected to a Python FastAPI backend. The backend stores ML traffic/risk observations in MongoDB and uses the existing Ruthvesh road graph/routing code for live routes and allocations.

## 1. Start MongoDB

Use local MongoDB (`mongodb://localhost:27017`) or MongoDB Atlas.

## 2. Start backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The backend seeds the initial dashboard data into MongoDB on first startup.

## 3. Start frontend

From the project root:

```bash
npm install
npm run dev
```

The frontend is configured for the live backend by default. `.env.example` contains:

```text
VITE_API_URL=http://localhost:8000
VITE_USE_MOCK=false
```

## 4. ML → MongoDB

Send each `prototype 2` / NovaRoute ML `traffic_data.json` observation to:

```text
POST http://localhost:8000/api/traffic
```

The backend stores the full observation in MongoDB and updates the corresponding junction's risk/congestion values.

## Data flow

```text
prototype 2 ML
      ↓
traffic_data.json
      ↓
POST /api/traffic
      ↓
MongoDB
      ↓
FastAPI backend
      ↓
Ruthvesh dynamic graph + Dijkstra
      ↓
final React frontend
```

No SQL database is used.
