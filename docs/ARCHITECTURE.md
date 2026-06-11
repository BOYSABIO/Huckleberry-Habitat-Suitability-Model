# Architecture

## Layers

```
CLI (src/main.py)
  ├── train  → pipelines/training_pipeline.py  → data + model.trainer + artifact registry
  └── infer  → inference/pipeline.py           → features + HabitatPredictor + reporting

API (future)
  └── POST /predict → model.artifact.HabitatPredictor   (features in → score out)
```

## Packages

| Package | Responsibility |
|---------|----------------|
| `src/model/` | Model **types**, training, artifact storage, and scoring |
| `src/model/registry.py` | **Model type catalog** — register `random_forest`, `ensemble`, etc. |
| `src/model/trainer.py` | Train any registered model type |
| `src/model/artifact_registry.py` | **Trained version catalog** — `registry.json` + `.joblib` files |
| `src/model/artifact.py` | `HabitatPredictor` — load artifacts and score feature matrices |
| `src/inference/pipeline.py` | **Location workflow** — lat/lon → env data → predictor |
| `src/inference/reporting.py` | Maps, CSVs, summaries, plots |
| `src/evaluation/` | Post-training analysis (feature importance) |
| `src/pipelines/` | Training orchestration + backward-compatible re-exports |

## predictor.py vs pipeline.py

These answer different questions:

- **`HabitatPredictor`** (`model/artifact.py`): “Here are the 12 climate features — what’s the suitability score?”
  - Fast, no network calls
  - Used by the future REST API

- **`InferencePipeline`** (`inference/pipeline.py`): “Here are coordinates — fetch GridMET/elevation/soil, build features, then score.”
  - Slow, heavy dependencies
  - Used by `python -m src.main infer`

The pipeline **calls** the predictor internally after feature extraction.

## Adding a new model type

1. Implement `src/model/implementations/your_model.py` with `fit`, `predict_proba`, `get_feature_importance`, `to_artifact`
2. Register it in `src/model/registry.py`
3. Training and registry pick it up via `settings.model.model_type`

## Canonical production artifact

Development inference uses `models/random_forest_improved.joblib` (raw sklearn RF from the notebook).
Pin `scikit-learn==1.6.1` when loading this file.
