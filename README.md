# Huckleberry Habitat Suitability Prediction

<p align="center">
  <img src="docs/assets/ezgif.com-video-to-gif-converter.gif" width="100%" alt="Huckleberry Detection Demo" />
</p>

Predict huckleberry (*Vaccinium membranaceum*) habitat suitability from GBIF occurrence records and environmental data (GridMET climate, elevation, soil pH). The pipeline handles ETL, pseudo-absence sampling, model training, and coordinate-based inference with maps and reports.

---

## Quick start

**Prerequisites:** Python 3.9+, [Conda](https://docs.conda.io/) (recommended), [Ollama](https://ollama.ai/) (geocoding fallback)

```bash
git clone <repository-url>
cd Capstone-Microsoft
conda env create -f environment.yml
conda activate Capstone-Microsoft
```

Verify with a fast training run on the small GBIF sample:

```bash
python -m src.main train --sample
```

**Train** from the cached model-ready snapshot (skips the ~5-hour ETL):

```bash
python -m src.main train --dataset hb
```

**Infer** at one or more coordinates (opens `outputs/maps/prediction_map.html` in a browser):

```bash
python -m src.main infer --coordinates 44.5 -116.5 --gridmet-date 2020-07-15
```

Use a specific trained model:

```bash
python -m src.main infer --coordinates 44.5 -116.5 --model models/your_model.joblib
```

**REST API** (feature-based scoring — see [REST API](#rest-api-feature-based) below):

```bash
pip install -r requirements-api.txt
uvicorn src.api.main:app --reload --port 8000
curl http://localhost:8000/health
```

Use any Python 3.10+ environment (venv, conda, etc.) — only `requirements-api.txt` is required for the API, not the full `requirements.txt` / `environment.yml` stack.

---

<details>
<summary><strong>Installation (detailed)</strong></summary>

### Python environment

**Conda (recommended)**

```bash
conda env create -f environment.yml
conda activate Capstone-Microsoft
```

**pip**

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Ollama (geocoding fallback)

```bash
# Windows
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

</details>

<details>
<summary><strong>Data layout</strong></summary>

### Active directories

| Path | Purpose |
|------|---------|
| `data/raw/occurrence.txt` | Full GBIF download — input for `train` |
| `data/raw/occurrence_sample.txt` | Small subset (~15 records) — `train --sample` |
| `data/processed/huckleberry_processed.csv` | Cleaned records before environmental enrichment |
| `data/enriched/huckleberry_enriched.csv` | Full training dataset with all features |
| `data/snapshots/` | Cached full datasets from the notebook era |
| `data/resources/manual_geocodes.json` | Curated locality → coordinate lookups |

### Snapshots

Produced by `docs/notebooks/Huckleberry.ipynb` so you can skip re-running expensive enrichment:

| File | Description |
|------|-------------|
| `snapshots/HB_PSEUDO_clean_elevation_soil.csv` | Complete enriched dataset (occurrences + pseudo-absences) |
| `snapshots/HB.csv` | Model-ready subset — used to train the current model (v13) |

Relationship: `HB_PSEUDO_*` → drop nulls, parse dates → `HB.csv`.

### Training presets (`--dataset`)

| Preset | File |
|--------|------|
| `hb` | `snapshots/HB.csv` — **recommended** |
| `hb_full` | `snapshots/HB_PSEUDO_clean_elevation_soil.csv` |

Or pass any path: `python -m src.main train --dataset path/to/file.csv`

`--dataset` and `--sample` are mutually exclusive.

### Archive

Notebook intermediates live in `archive/data/notebook/`. Not used by `src/`.

</details>

<details>
<summary><strong>Training</strong></summary>

```bash
# Full ETL from raw GBIF (several hours)
python -m src.main train

# Quick smoke test (~15 records)
python -m src.main train --sample

# Skip ETL — train from snapshot
python -m src.main train --dataset hb
python -m src.main train --dataset data/snapshots/HB.csv

# Model type (default: random_forest)
python -m src.main train --dataset hb --model-type random_forest
python -m src.main train --dataset hb --model-type ensemble
```

**What happens**

1. Load GBIF data (or CSV when `--dataset` is set)
2. Clean, filter, geocode
3. Generate pseudo-absences (training from raw data only)
4. Extract GridMET, elevation, and soil features
5. Train and register model in `models/registry.json`
6. Write processed/enriched CSVs and feature-importance reports

**Model types**

| Type | Notes |
|------|-------|
| `random_forest` | Default; fast and interpretable |
| `ensemble` | XGBoost + BernoulliNB stacking (requires TPOT) |

</details>

<details>
<summary><strong>Inference</strong></summary>

```bash
# Multiple coordinates (lat lon pairs)
python -m src.main infer --coordinates 44.5 -116.5 44.6 -116.4

# Season / climate date for GridMET
python -m src.main infer --coordinates 44.5 -116.5 --gridmet-date 2020-07-15

# Custom model and confidence threshold
python -m src.main infer --coordinates 44.5 -116.5 \
  --model models/huckleberry_model_v13.joblib \
  --confidence-threshold 0.6

# Skip map generation
python -m src.main infer --coordinates 44.5 -116.5 --no-map
```

**Defaults**

- Model: registry `current` → `huckleberry_model_v13` (override with `--model`)
- GridMET date: latest available (override with `--gridmet-date YYYY-MM-DD`, range ~1979–2020)
- Confidence threshold: `0.8` — counts as “suitable habitat” in summary stats; map shows **all** points color-coded (green / orange / red)

**What happens**

1. Validate coordinates
2. Fetch GridMET, elevation, and soil for each point
3. Score with `HabitatPredictor`
4. Write CSV, optional HTML map, JSON summary, and confidence plot

</details>

<details>
<summary><strong>REST API (feature-based)</strong></summary>

The FastAPI service scores a single 12-feature row (same columns as `HB.csv` / `PredictRequest`). It does **not** fetch GridMET or elevation — clients supply prepared features.

### Install and run

```bash
pip install -r requirements-api.txt
uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + loaded `model_version` |
| `POST` | `/predict` | Returns `probability` and tree-based `confidence_interval` |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2020, "season_num": 2, "elevation": 1200.0, "soil_ph": 5.5,
    "air_temperature": 15.2, "precipitation_amount": 2.1,
    "specific_humidity": 0.008, "relative_humidity": 65.0,
    "mean_vapor_pressure_deficit": 0.5, "potential_evapotranspiration": 3.2,
    "surface_downwelling_shortwave_flux_in_air": 250.0, "wind_speed": 2.5
  }'
```

### Model loading priority

The API loads models in this order:

1. **`MODEL_PATH`** — path to a `.joblib` file (used by CI/tests via `tests/fixtures/test_model.joblib`)
2. **MLflow** — when `MLFLOW_TRACKING_URI` is set (see [MLflow](#mlflow-model-tracking--registry) below)
3. **Local registry** — `models/registry.json` → `current` version

</details>

<details>
<summary><strong>MLflow (model tracking & registry)</strong></summary>

Training can log runs and register models in MLflow when a tracking server is configured. The API can load the production model from the registry instead of local `registry.json`.

### Local tracking server (MLflow 3)

Use SQLite for the backend store (the plain `mlruns/` file store is unreliable on MLflow 3):

```bash
pip install -r requirements-api.txt

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"
```

In another terminal:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.main train --dataset hb
```

Each training run logs to experiment **`huckleberry-training`** and registers model **`huckleberry-habitat`**. If `MLFLOW_TRACKING_URI` is not set, training still updates the local `models/registry.json` only.

### Promote a model for the API

In the MLflow UI (http://localhost:5000):

1. Open **Models** → **`huckleberry-habitat`**
2. Select the version you want to serve
3. Assign alias **`production`** (use **Aliases**, not legacy “Promote to Production” stages)

Default URI used by the API when tracking is enabled:

```text
models:/huckleberry-habitat@production
```

### Run the API against MLflow

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_MODEL_URI=models:/huckleberry-habitat@production   # optional; this is the default
uvicorn src.api.main:app --reload --port 8000
curl http://localhost:8000/health
```

`/health` should report a `model_version` like `mlflow:models:/huckleberry-habitat@production`.

### Dev cleanup (optional)

Remove junk experiments from early setup (does not delete registered models):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/mlflow_cleanup.py --dry-run
python scripts/mlflow_cleanup.py --delete Default lesson-1-smoke
```

Local MLflow files (`mlflow.db`, `mlartifacts/`, `mlruns/`) are gitignored.

</details>

<details>
<summary><strong>Docker (API + MLflow)</strong></summary>

**Files:** `Dockerfile` builds the API image; `docker-compose.yml` orchestrates API + MLflow together. Compose uses `build: .`, so you need both — compose is not a replacement for the Dockerfile.

On Windows, training logs artifact paths as `file:///C:/...`. The compose stack runs `scripts/mlflow_docker_prepare.py` before MLflow starts to rewrite those paths for Linux containers, and mounts `mlartifacts/` into the API container so it can read model files.

### Recommended: docker compose

Runs API and MLflow on the same Docker network (avoids `host.docker.internal` issues on Windows).

**Prerequisites:** `mlflow.db`, `mlartifacts/`, and alias **`production`** on **`huckleberry-habitat`** (train with `MLFLOW_TRACKING_URI` set first).

```powershell
docker compose up --build
```

- API: http://localhost:8000/docs  
- MLflow UI: http://localhost:5000  

```powershell
curl http://localhost:8000/health
```

First startup can take **30–60 seconds** while the API loads the model from MLflow.

Stop with `Ctrl+C`, or `docker compose down` in another terminal.

### Standalone `docker run` (MLflow on host)

Start MLflow **before** the container, from repo root:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000 --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"
```

Verify on the host: open http://localhost:5000

Then (stop local uvicorn so port 8000 is free):

```powershell
docker build -t huckleberry-api .
docker run --rm -p 8000:8000 huckleberry-api
```

If you see **`Connection refused`** to `host.docker.internal:5000`, MLflow is not running or not listening on `0.0.0.0:5000`.

If startup **hangs** at `Waiting for application startup`, MLflow may be downloading a large artifact — wait up to a minute, or use `docker compose` instead.

</details>

<details>
<summary><strong>Tests & development</strong></summary>

From repo root, with Python 3.10+ and API deps installed:

```bash
pip install -r requirements-api.txt
python -m pytest tests/ -v
```

CI runs the same tests plus `docker compose config` and `docker build`. The API tests load `tests/fixtures/test_model.joblib` via `MODEL_PATH` (see `tests/conftest.py`).

Regenerate the fixture after changing the model schema or scikit-learn pin:

```bash
python scripts/build_test_fixture.py
```

</details>

<details>
<summary><strong>Outputs</strong></summary>

| Location | Contents |
|----------|----------|
| `outputs/predictions/` | `inference_predictions.csv`, timestamped `top_predictions_*.csv` |
| `outputs/maps/` | `prediction_map.html` — interactive Folium map |
| `outputs/summaries/` | `inference_summary_*.json`, `confidence_plot_*.png` |
| `outputs/feature_importance/` | CSV + plot per training run |
| `data/processed/`, `data/enriched/` | Pipeline CSV outputs |
| `models/` | `.joblib` artifacts + `registry.json` |
| `logs/` | `pipeline.log` |

</details>

<details>
<summary><strong>Architecture</strong></summary>

`src/main.py` is a thin CLI — it parses arguments and dispatches to orchestrators.

```
src/
  main.py              CLI
  config/              Settings and presets
  data_load/           I/O only (CSV, GBIF)
  data_preprocess/     Cleaning, filtering, geocoding
  data_validation/     Schema checks at pipeline boundaries
  features/            Environmental, temporal, pseudo-absence sampling
  training/            Training orchestration
  inference/           Coordinate inference + maps/reports
  model/               Registry, training, version store, predictor
  evaluation/          Feature importance
  utils/               Logging, data versioning
```

### Training flow

```mermaid
flowchart LR
    main[main.py] --> train[training/pipeline.py]
    train --> io[data_load/loader.py]
    train --> prep[data_preprocess]
    train --> valid[data_validation]
    train --> feat[features]
    train --> mdl[model/trainer.py]
    mdl --> store[model/store.py]
```

### Inference flow

```mermaid
flowchart LR
    main[main.py] --> infer[inference/pipeline.py]
    infer --> feat[features/environmental.py]
    infer --> score[model/predictor.py]
    infer --> report[inference/reporting.py]
```

### Model package

| Module | Role |
|--------|------|
| `model/registry.py` | `@register("name")` + `create_model()` |
| `model/implementations/` | Algorithm classes (`fit`, `predict_proba`) |
| `model/features.py` | Feature matrix column selection |
| `model/trainer.py` | Training entry point |
| `model/store.py` | `registry.json` + versioned `.joblib` files |
| `model/predictor.py` | Load models + `HabitatPredictor` scoring |
| `model/mlflow_logging.py` | Log training runs and register models in MLflow |
| `model/mlflow_config.py` | Shared MLflow experiment/model naming constants |

Loading and scoring live in `model/`, not `inference/`. Inference orchestrates coordinates → environmental features → predictor.

| Question | Module |
|----------|--------|
| Given features, what's the score? | `model/predictor.py` |
| Given lat/lon, what's the score? | `inference/pipeline.py` |

### Adding a new model type

1. Implement `src/model/implementations/your_model.py` with `@register("your_name")`
2. Import it in `src/model/implementations/__init__.py`
3. Train with `--model-type your_name`

> `docs/` holds research artifacts (notebooks, notes, meeting notes). `archive/` holds legacy scripts and notebook-era data. The production pipeline in `src/` is independent.

</details>

<details>
<summary><strong>Repository layout</strong></summary>

```
Capstone-Microsoft/
├── data/                   # Raw, processed, enriched, and snapshot datasets
├── docs/                   # Research and project documentation
│   ├── notebooks/          # Pre-pipeline exploration (e.g. Huckleberry.ipynb)
│   ├── notes/              # Planning and exploration write-ups
│   ├── meeting_notes/
│   ├── references/
│   ├── legacy/             # Archived July 2025 runs (models + inference outputs)
│   └── assets/             # README images and GIFs
├── models/                 # Trained .joblib files and registry.json
├── outputs/                # Predictions, maps, summaries, feature importance
├── src/                    # Production pipeline (CLI, training, inference)
├── archive/                # Legacy scripts and notebook-era intermediates
├── logs/
└── tests/
```

</details>

<details>
<summary><strong>Configuration</strong></summary>

Settings: `src/config/settings.py` and `src/config/environments.py`.

| Mode | Command | Raw input | `n_estimators` |
|------|---------|-----------|----------------|
| Full | `train` | `data/raw/occurrence.txt` | 200 |
| Sample | `train --sample` | `data/raw/occurrence_sample.txt` | 50 |

All runs write to the same paths: `data/processed/huckleberry_processed.csv` and `data/enriched/huckleberry_enriched.csv`.

**Pseudo-absences:** ratio 3:1, 5 km buffer from real occurrences.

**Environmental features:** 8 GridMET variables, elevation (Open-Elevation), soil pH (SoilGrids), temporal/season columns.

</details>

<details>
<summary><strong>Testing</strong></summary>

```bash
pytest tests/
pytest tests/test_basic.py
pytest --cov=src tests/
```

</details>

<details>
<summary><strong>Performance & timing</strong></summary>

| Stage | Typical time |
|-------|----------------|
| Full ETL + enrichment | ~5+ hours |
| Train from `hb` snapshot | ~5–15 min |
| Inference per coordinate | ~30–60 s |
| Model accuracy (snapshot) | ~85–95% |

Long GridMET runs can fail if Planetary Computer signed URLs expire mid-run — prefer `--dataset hb` for training unless you need a fresh ETL.

</details>

<details>
<summary><strong>Troubleshooting</strong></summary>

| Issue | What to try |
|-------|-------------|
| GridMET / Planetary Computer errors | Check network; reinstall `pystac-client` and `planetary-computer`; use `--dataset hb` to skip ETL |
| Low inference confidence | Try a different `--gridmet-date`; check elevation API (504s fill elevation with 0) |
| Model not found | Pass `--model path/to/file.joblib` or check `models/registry.json` |
| MLflow connection refused | Confirm server is running and `MLFLOW_TRACKING_URI` matches (e.g. `http://localhost:5000`) |
| API loads wrong model | Check env: `MODEL_PATH` overrides MLflow; MLflow needs both `MLFLOW_TRACKING_URI` and alias `@production` |
| Geocoding failures | See `data/resources/manual_geocodes.json` |
| Memory pressure | Use `train --sample` or `--dataset hb` |

Logs: `logs/pipeline.log`

</details>

<details>
<summary><strong>Legacy runs (July 2025)</strong></summary>

The first successful capstone demos (v9/v10 dev models, inference on 17 coordinates, confidence plots) are archived under [`docs/legacy/`](docs/legacy/README.md) — not used by the active pipeline.

Includes the notebook-era `random_forest_improved.joblib`, feature-importance CSVs, top predictions, and inference summaries from 2025-07-05.

</details>

<details>
<summary><strong>Data sources & acknowledgments</strong></summary>

- [GBIF — *Vaccinium membranaceum*](https://www.gbif.org/species/9060377)
- [Microsoft Planetary Computer — GridMET](https://planetarycomputer.microsoft.com/dataset/gridmet)
- [Open-Elevation API](https://api.open-elevation.com/)
- [SoilGrids](https://www.isric.org/explore/soilgrids)

</details>
