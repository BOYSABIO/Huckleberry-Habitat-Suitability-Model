# Weekly Task — 2026-06-05

**Project:** Huckleberry Habitat Suitability Model
**Type:** Mixed
**Tool:** Both (Cursor for experiment code, Cowork for findings write-up)

## Task

Dockerize the Huckleberry model and deploy it as a REST API on the homelab. Wrap the trained Random Forest model in a FastAPI endpoint that accepts climate feature inputs (same schema as training features) and returns a habitat suitability probability + confidence interval. Write a `docker-compose.yml`, build the image, and deploy the service on a Proxmox LXC in the homelab. Update the README with an API usage section and a curl example.

## Deliverable

A running service at `http://homelab-ip:8000/predict` with a `/predict` POST endpoint and `/docs` Swagger page, a `docker-compose.yml` at the repo root, and a README section documenting how to deploy and query it — making the model callable as infrastructure rather than a notebook artifact.

## Context for Cursor

The Huckleberry repo lives at `PROJECTS/Capstone-Microsoft/` (also at https://github.com/BOYSABIO/Huckleberry-Habitat-Suitability-Model). The trained Random Forest model is the primary artifact — locate the model serialization code (likely pickle or joblib) and the feature schema from the training pipeline. Goal: (1) wrap the model in a FastAPI app with a `/predict` endpoint that accepts a JSON body matching the training feature schema and returns `{"probability": float, "confidence_interval": [float, float]}`; (2) write a `Dockerfile` that installs dependencies and serves the app; (3) write a `docker-compose.yml`; (4) test locally, then deploy to a Proxmox LXC on the homelab. Success condition: `docker-compose up` produces a working service that returns valid predictions. The README update is a Cowork task — hand off the API description and a sample request/response after Cursor ships the service.

---

## What are we serving?

The repo has **two different inference concepts**. We need to pick one for this API.

### Option A — Feature-based API (this task)

**User sends:** pre-computed climate/environmental features (same columns the model was trained on).

**API does:** validate JSON → scale features → run Random Forest → return probability + confidence interval.

**Example request body:**
```json
{
  "year": 2020,
  "season_num": 2,
  "elevation": 1200.0,
  "soil_ph": 5.5,
  "air_temperature": 15.2,
  "precipitation_amount": 2.1,
  "specific_humidity": 0.008,
  "relative_humidity": 65.0,
  "mean_vapor_pressure_deficit": 0.5,
  "potential_evapotranspiration": 3.2,
  "surface_downwelling_shortwave_flux_in_air": 250.0,
  "wind_speed": 2.5
}
```

**Example response:**
```json
{
  "probability": 0.73,
  "confidence_interval": [0.55, 0.89]
}
```

**Pros:** Simple, fast, small Docker image, no external APIs, matches the task spec exactly.

**Cons:** Caller must already have the feature values (from your training pipeline, another service, or manual lookup).

### Option B — Location-based API (existing `InferencePipeline`)

**User sends:** lat/lon (and optionally a date).

**API does:** fetch GridMET climate data, elevation (Open-Elevation), soil pH (SoilGrids) → build features → predict.

**Pros:** Easier for end users — “is this spot good?” with just coordinates.

**Cons:** Heavy dependencies (Planetary Computer, xarray, network calls), slower, fragile offline, much bigger Docker image. **Not what this task asks for.**

### Decision

**Go with Option A for this task.** The task explicitly says *“accepts climate feature inputs (same schema as training features).”*

The current model (`registry.json` → `"current"`) expects these 12 features:

| Feature | Description |
|---------|-------------|
| `year` | Year of observation |
| `season_num` | 0=winter, 1=spring, 2=summer, 3=fall |
| `elevation` | Meters |
| `soil_ph` | Soil pH |
| `air_temperature` | GridMET variable |
| `precipitation_amount` | GridMET variable |
| `specific_humidity` | GridMET variable |
| `relative_humidity` | GridMET variable |
| `mean_vapor_pressure_deficit` | GridMET variable |
| `potential_evapotranspiration` | GridMET variable |
| `surface_downwelling_shortwave_flux_in_air` | GridMET variable |
| `wind_speed` | GridMET variable |

Option B can be a **future enhancement** (e.g. `/predict-from-location`) — out of scope for now.

---

## Why v1 is feature-based (and how it scales later)

v1 is intentionally a **thin model service**, not a full “lat/lon → answer” product API.

**Why we ship it this way first:**
- Matches the task spec (feature JSON in → probability out).
- Small Docker image, fast inference, no external API dependencies per request.
- Easier to test, deploy on homelab, and debug on a fresh clone.
- Proves the trained artifact is **callable as infrastructure** — the core deliverable.

**Why that’s still good architecture (not a shortcut):**
Production ML systems usually separate **feature engineering** from **model scoring**. Training already builds features upstream; the model only ever sees the final column vector. v1 mirrors that boundary.

**How it scales (v2+):**

```
v1 (now)     POST /predict              features → score
v2           POST /predict-from-location  lat/lon → [feature pipeline] → score
v3           Split into two services      feature-service + model-service
             + caching                    same lat/lon/date → skip refetch
```

- **v2:** Add a second endpoint that wraps existing `InferencePipeline` logic internally, then calls the same prediction code as `/predict`. Users get convenience; model logic stays in one place.
- **v3:** Run feature extraction as its own container/job (batch or async), cache results by `(lat, lon, date)`, keep the model container tiny and always-on.

The location-based path is **more useful to casual users** — it’s the right long-term UX. It’s just the wrong **first** deploy because it couples prediction to GridMET, elevation APIs, heavy deps, and network reliability. Build the stable core first, then wrap it.

---

## Getting started on a fresh machine

You cloned the repo and haven’t installed dependencies yet — **that’s fine**. Do **not** `pip install -r requirements.txt` for this API task unless you also plan to run the full training/inference pipeline. That file pulls geospatial stacks, Planetary Computer, TPOT, etc. Most of it is irrelevant to v1.

### Step 1 — Python version (important)

The project was developed on **Python 3.9** (see `environment.yml`). Your system Python may be newer (e.g. 3.14). Bleeding-edge Python often breaks `scikit-learn` wheels.

**Recommended:** create a venv with **Python 3.11** (or 3.10–3.12):

```bash
# if you have python3.11 installed
python3.11 -m venv .venv
source .venv/bin/activate
```

If you only have system `python3`, try it — but if `pip install scikit-learn` fails, install `python3.11` via your distro and use that.

### Step 2 — Minimal install (API / Phase 1 only)

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn joblib
```

That’s enough for Phase 1 (load model + predict in a REPL). Add `fastapi uvicorn[standard]` in Phase 2.

Later you’ll create `requirements-api.txt` with just these packages for Docker. The full `requirements.txt` stays for the existing pipeline notebooks.

### Step 3 — Model artifact on disk (check this before Phase 1)

`registry.json` lists many model versions, but large `.joblib` files are **gitignored**. On a fresh clone you may only have:

- `models/registry.json` (metadata — in git)
- `models/random_forest_improved.joblib` (one legacy model — in git)

The `"current"` entry in `registry.json` may point at a file that **isn’t on this machine** (e.g. `huckleberry_model_dev_v10_...joblib` from a training run on another box).

**Before Phase 1, pick one path:**

| Situation | What to do |
|-----------|------------|
| You have the v10 `.joblib` on another machine | `scp` it into `models/` on the laptop |
| You only have `random_forest_improved.joblib` | Use that file for v1 API dev; inspect its feature schema after loading (may differ from registry v10) |
| You need the registry “current” model | Copy the matching `.joblib` or re-train locally |

For the API task, you need **one working `.joblib`**, not the entire registry history. Document which file the API loads in your Dockerfile/README.

### Step 4 — Run from repo root

Imports use `from src....` — always activate the venv and run commands from the repo root:

```bash
cd /path/to/Huckleberry-Habitat-Suitability-Model
source .venv/bin/activate
python -c "from src.models.registry import ModelRegistry; print('import ok')"
```

### What you can ignore for now

- `environment.yml` / conda — only needed for full pipeline reproduction
- Full `requirements.txt` — Phase 4+ Docker uses a slim API requirements file
- Training data, GridMET, notebooks — not needed for v1 API

---

## Implementation plan

Work through these phases in order. Do not move to Docker until local prediction works.

### Phase 0 — Prerequisites

- [ ] Create `.venv` with Python 3.11 (or 3.10–3.12) — avoid relying on bleeding-edge system Python
- [ ] `pip install numpy pandas scikit-learn joblib` only (not full `requirements.txt`)
- [ ] Confirm which `.joblib` you will actually serve (see **Getting started** above)
- [ ] Confirm that file exists under `models/` on this machine
- [ ] Note its `feature_names` — that becomes your API contract (may differ from registry `"current"` if files are missing)

### Phase 1 — Understand the artifact (≈30 min)

**Goal:** Load the model in a REPL and produce one prediction before writing any API code.

1. Read `src/models/registry.py` — how `ModelRegistry.load_model()` works
2. Read `src/models/pipeline.py` — `RandomForestPredictor` wraps a `StandardScaler` + sklearn `RandomForestClassifier`
3. Smoke test in Python:

```python
from src.models.registry import ModelRegistry
import pandas as pd

registry = ModelRegistry("models/")
data = registry.load_model()
predictor = data["model"]

# One row — use real-ish values, keys must match predictor.feature_names
row = {name: 0.0 for name in predictor.feature_names}
df = pd.DataFrame([row])
print(predictor.predict(df))
```

**Done when:** No errors loading or predicting.

**Key detail:** Inference is not raw sklearn on JSON. Features must be scaled the same way as training (`predictor.scaler` + inner `predictor.model`).

### Phase 2 — FastAPI app (local, no Docker)

**Goal:** `http://localhost:8000/docs` shows a working `/predict` endpoint.

1. Add deps: `fastapi`, `uvicorn[standard]` (consider a slim `requirements-api.txt` for Docker later)
2. Create `src/api/` with:
   - `main.py` — FastAPI app, load model once at startup
   - `schemas.py` — Pydantic request/response models matching the 12 features
   - `predictor.py` — prediction logic (reuse registry + scaling)
3. Implement endpoints:
   - `POST /predict` → `{"probability": float, "confidence_interval": [float, float]}`
   - `GET /health` — cheap sanity check for deploy
4. Run locally:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Done when:** Swagger UI accepts JSON and returns a valid probability.

### Phase 3 — Confidence interval

**Goal:** Response includes a meaningful interval, not just a point estimate.

The existing pipeline uses “confidence” as probability thresholds (0.6, 0.8). This task wants a numeric interval `[low, high]`.

**Approach (Random Forest):** use per-tree probabilities:
- Point estimate = mean across trees
- Interval = e.g. 2.5th and 97.5th percentiles across trees

Document in code what the interval means (“uncertainty across trees in the forest”).

**Done when:** Response shape matches `{"probability": ..., "confidence_interval": [..., ...]}`.

### Phase 4 — Dockerize

**Goal:** `docker run -p 8000:8000 huckleberry-api` works.

1. `Dockerfile` at repo root:
   - `python:3.11-slim` base
   - Install slim requirements only (not full geospatial stack)
   - `COPY` `src/` and `models/` (registry.json + current `.joblib`)
   - `CMD` uvicorn on port 8000
2. Watch out for:
   - Paths are relative to `/app` — registry expects `models/`
   - `.dockerignore` must **not** exclude the `.joblib` you need
3. Test:

```bash
docker build -t huckleberry-api .
docker run -p 8000:8000 huckleberry-api
curl http://localhost:8000/health
```

**Done when:** Container returns predictions via curl.

### Phase 5 — docker-compose.yml

**Goal:** `docker compose up` is the one-command deploy.

1. `docker-compose.yml` at repo root — one `api` service, port `8000:8000`, `restart: unless-stopped`
2. Test: `docker compose up --build`

**Done when:** Same as Phase 4 but via compose.

### Phase 6 — Homelab deploy (Proxmox LXC)

**Goal:** `curl http://<homelab-ip>:8000/predict` works from another machine.

1. Create LXC (Debian/Ubuntu), install Docker
2. Clone repo or copy image to the box
3. Ensure model artifacts are present (`registry.json` + `.joblib`)
4. `docker compose up -d`
5. Open port 8000 (Proxmox firewall + LXC network)

**Done when:** Service reachable at homelab IP.

### Phase 7 — README handoff (Cowork)

After the service works locally or on homelab, hand Cowork:

- Base URL and `/predict` contract
- Sample request + response JSON
- `curl` example
- Deploy instructions (`docker compose up --build`)

---

## Architecture (Option A)

```
Client  →  POST /predict (JSON features)  →  FastAPI
                                              ↓
                                         ModelRegistry
                                              ↓
                                         .joblib (RandomForestPredictor)
                                              ↓
                              probability + confidence_interval  →  Client
```

## What not to do

- Do not wire up full `InferencePipeline` for this task (no GridMET/elevation/soil fetching in v1)
- Do not install all of `requirements.txt` in Docker unless you want a huge image
- Do not hand-edit `registry.json` — use the model marked `"current"`

## Debug checkpoints

| Symptom | Likely cause |
|---------|----------------|
| REPL load fails | Missing `.joblib` or wrong path |
| API 422 | Request JSON doesn't match `feature_names` |
| API 500 on predict | Wrong column order or missing scaling |
| Docker works locally, not on LXC | Model files not copied into image |
| Can't reach homelab | Firewall/networking, not the model |

## Current step

**→ Phase 0:** Set up a minimal venv, confirm which model file you have on disk, then load it in a REPL (Phase 1).
