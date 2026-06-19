# Future tasks (after TASK.md)

Complete [TASK.md](TASK.md) first (API + Docker + MLflow + **manual** homelab deploy + GitHub Actions CI). Then consider, in roughly this order:

---

## Architecture (how serving, training, and updates fit together)

Read this before homelab deploy and before automating CD. Full step-by-step for Proxmox LXC: [docs/homelab-deploy.md](docs/homelab-deploy.md).

### What the Docker stack serves

The **API container does not bake in a model**. At startup it loads from MLflow:

```text
MLFLOW_MODEL_URI=models:/huckleberry-habitat@production
```

Whatever version has the **`production`** alias in the MLflow registry is loaded once into memory. `/health` reports e.g. `mlflow:models:/huckleberry-habitat@production`.

**Loading priority** (see `src/model/predictor.py`):

1. `MODEL_PATH` — explicit `.joblib` (CI/tests only)
2. **MLflow** — when `MLFLOW_TRACKING_URI` is set (Docker compose default)
3. `models/registry.json` → `current` — local CLI fallback without MLflow

**Persistence on disk** (gitignored; must exist on the homelab host):

| Path | Role |
|------|------|
| `mlflow.db` | Registry metadata, run history, `@production` alias |
| `mlartifacts/` | Model files (`.joblib`, MLmodel metadata) |

Compose mounts these into the MLflow and API containers. The **Docker image** is only code + slim API dependencies.

### Serving vs training (two different jobs)

| Job | Where it runs | What it does |
|-----|----------------|--------------|
| **Serving** | `docker compose up` (API + MLflow containers) | `/predict`, MLflow UI on ports 8000 / 5000 |
| **Training** | Host OS with full env (`requirements.txt` / conda) | `python -m src.main train --dataset hb` |

Training is **not** exposed over HTTP. You run it via SSH on the LXC (or on your PC with `MLFLOW_TRACKING_URI` pointing at homelab). The slim API image does not include GridMET/geopandas.

After training: assign **`production`** in the MLflow UI, then **`docker compose restart api`** so the API picks up the new model text.

### Homelab network exposure

From another machine on your LAN:

| URL | Purpose |
|-----|---------|
| `http://<homelab-ip>:8000/docs` | Swagger — call `/predict` |
| `http://<homelab-ip>:8000/health` | Model loaded? Which version? |
| `http://<homelab-ip>:5000` | MLflow UI — promote models |

MLflow 3.x requires your LXC LAN address in `MLFLOW_ALLOWED_HOSTS` (local `.env` on the server — see [homelab deploy](docs/homelab-deploy.md#5b-mlflow-ui-from-your-pc-invalid-host-header)). The API on :8000 does not.

SSH is for admin (git pull, compose, training), not for end users.

### Manual update flow (Phase 7 — what you do today)

**Code / config changes** (Dockerfile, compose, API code, README):

```bash
# On your PC: develop, test, merge to main

# On the LXC (SSH):
cd ~/Capstone-Microsoft   # or your clone path
docker compose down
git pull origin main
./scripts/homelab_start.sh
curl http://localhost:8000/health
```

That is the correct manual “deploy update” loop until homelab CD (§1 below) automates it.

**Model-only changes** (new training run, no repo changes):

1. Train with `MLFLOW_TRACKING_URI` pointing at homelab MLflow (or train locally and copy `mlflow.db` + `mlartifacts/`).
2. Set **`production`** alias in MLflow UI.
3. `docker compose restart api` — no `git pull` required.

**What git pull does *not* update:** `mlflow.db` and `mlartifacts/` (gitignored). Copy those with `scp`/`rsync` when seeding a fresh LXC or syncing models from your PC.

### Enterprise analogy (miniature)

| This project | Typical enterprise |
|--------------|-------------------|
| MLflow `@production` | Model registry stage |
| `mlflow.db` + `mlartifacts/` | MLflow + S3/artifact store |
| FastAPI in Docker | Managed serving endpoint |
| Manual SSH + `git pull` | CI/CD deploy pipeline (FUTURE_TASKS §1) |
| Restart API after alias change | Rolling deploy / model reload hook |

---

## 1. Homelab CD (automated deploy) — recommended next

**Goal:** Merge to `main` (or tag a release) → homelab Proxmox LXC pulls and restarts the stack without manual SSH.

TASK.md Phase 7 is manual (`ssh` → `docker compose up`). This automates that.

### Options (pick one)

| Approach | Pros | Cons |
|----------|------|------|
| **Self-hosted GitHub Actions runner** on the LXC | Native `docker compose` on the box; no inbound SSH from internet | Runner maintenance; runner must stay online |
| **SSH deploy job** (`appleboy/ssh-action` or similar) | Simple; no permanent runner | Store homelab SSH key as GitHub secret; expose SSH carefully |
| **GHCR + pull on LXC** | Build/push image in CI; LXC only `docker pull` + restart | Still need a trigger on the LXC (webhook, cron, or runner) |

### Typical flow

```
push to main  →  CI builds + tests + pushes image to GHCR
              →  deploy job SSHs to LXC (or self-hosted runner on LXC)
              →  docker compose pull && docker compose up -d
              →  smoke curl /health on homelab IP
```

### Checklist

- [ ] Container registry (GitHub Container Registry `ghcr.io`)
- [ ] GitHub secrets: `HOMELAB_HOST`, `HOMELAB_SSH_KEY`, optional `HOMELAB_USER`
- [ ] `.github/workflows/deploy.yml` — `workflow_dispatch` first, then `push` to `main` when stable
- [ ] Document rollback (`docker compose down` + previous image tag)
- [ ] Optional: Tailscale/WireGuard so homelab is not on the public internet

---

## 2. CI hardening

Expand beyond TASK.md’s minimal `pytest` + `docker build`:

- [ ] Coverage threshold on `src/`
- [ ] Lint (`ruff` or `flake8`) on PRs
- [ ] API contract test with pinned fixture JSON
- [ ] Fail PR if `docker compose config` is invalid
- [ ] Optional: build matrix (Python 3.10 / 3.11)

---

## 3. DVC — dataset versioning

When re-running full ETL or sharing large datasets across machines:

- Version `data/raw`, processed/enriched outputs, snapshots
- Remote storage (S3, Azure, or local NAS)
- `dvc repro` for pipeline stages

Pair with MLflow (models) — DVC owns data blobs, MLflow owns experiments.

---

## 4. W&B — experiment UI (optional)

Only if MLflow dashboards are not enough for comparing training runs. Can supplement or replace experiment **tracking** UI; keep MLflow registry for production promotion unless you standardize on W&B Model Registry.

---

## 5. Location-based API (v2)

`POST /predict-from-location` wrapping `InferencePipeline` (lat/lon → GridMET/elevation/soil → score). See TASK.md “v2+” notes. Heavier Docker image; not for v1.

---

## Summary order

| Priority | Item |
|----------|------|
| Now | [TASK.md](TASK.md) — including Phase 0b CI + manual homelab |
| Next | **Homelab CD** (this file §1) |
| Then | CI hardening → DVC → W&B (if needed) → location API |
