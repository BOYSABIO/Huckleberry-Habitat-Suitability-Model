# Future tasks (after TASK.md)

Complete [TASK.md](TASK.md) first (API + Docker + MLflow + **manual** homelab deploy + GitHub Actions CI). Then consider, in roughly this order:

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
