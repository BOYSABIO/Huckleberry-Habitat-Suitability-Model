# Legacy runs (July 2025)

These artifacts are from the **original capstone inference demos** — the first successful end-to-end runs before the refactored `src/` pipeline and `HB.csv` training snapshot.

They are kept for reference and documentation, not for active training or deploy.

## Models (`models/`)

| File | Notes |
|------|--------|
| `random_forest_improved.joblib` | Notebook-era model trained on `data/snapshots/HB.csv`; previously the default inference artifact |
| `huckleberry_model_dev_v9_20250705_132103.joblib` | Dev pipeline run on the small GBIF sample |
| `huckleberry_model_dev_v10_20250705_141054.joblib` | Follow-up dev run; paired with the second inference demo below |

Registry metadata for v9/v10: `models/registry_2025-07-05.json`

## Inference run outputs (`runs/2025-07-05/`)

Two inference sessions on **17 Idaho-area coordinates** using the early feature-based predictor:

| Run | Summary | Confidence plot |
|-----|---------|-----------------|
| 13:44 | `summaries/inference_summary_20250705_134424.json` | `summaries/confidence_plot_20250705_134424.png` |
| 14:16 | `summaries/inference_summary_20250705_141642.json` | `summaries/confidence_plot_20250705_141643.png` |

Highlights from the 14:16 run: **100% suitable** predictions, **~82% average confidence**, top spot at `(44.7, -116.3)` at 98% probability.

Also preserved:

- `predictions/top_predictions_20250705_*.csv` — high-confidence coordinate lists per run
- `feature_importance/feature_importance_vhuckleberry_model_dev_v9_*.csv` and `v10_*.csv`

## Active pipeline today

Train and infer with the current CLI:

```bash
python -m src.main train --dataset hb
python -m src.main infer --coordinates 44.5 -116.5 --model models/huckleberry_model_v13_20260612_111519.joblib
```

Current registry: `models/registry.json` → `huckleberry_model_v13_20260612_111519` (~95% accuracy on 6,240 records).
