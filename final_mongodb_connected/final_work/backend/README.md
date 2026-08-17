# NovaRoute.AI MongoDB Backend

This backend connects the final React UI to MongoDB and the existing Ruthvesh routing project.

## Start MongoDB

Run a local MongoDB server or use MongoDB Atlas.

## Install

```bash
cd backend
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` or export the variables in your shell.

## Run

```bash
uvicorn app:app --reload --port 8000
```

The API is available at `http://localhost:8000`.

The first startup seeds MongoDB with the dashboard's initial junction/officer data. The ML pipeline can then POST its `traffic_data.json` records to `POST /api/traffic`. Those observations are stored in MongoDB and their `risk_score` / `congestion_factor` become available to the final UI and dynamic Ruthvesh routing.
