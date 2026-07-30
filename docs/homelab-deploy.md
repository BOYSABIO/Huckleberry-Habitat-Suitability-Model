# Homelab deploy (Proxmox LXC) — Issue #11

Manual deploy of the API + MLflow stack to a Proxmox LXC. Automated CD prep is tracked in [Issue #22](https://github.com/BOYSABIO/Huckleberry-Habitat-Suitability-Model/issues/22) and the workflow itself in [Issue #23](https://github.com/BOYSABIO/Huckleberry-Habitat-Suitability-Model/issues/23).

For architecture (which model loads, serving vs training, update flows), see [ROADMAP.md](../ROADMAP.md).

---

## What you are building

```text
[Your LAN]
    │
    ├── http://<LXC-IP>:8000  →  FastAPI (/predict, /docs)
    ├── http://<LXC-IP>:5000  →  MLflow UI (promote @production)
    │
    └── SSH → LXC host
              ├── git clone (repo)
              ├── mlflow.db + mlartifacts/ (model store)
              └── docker compose (see homelab_start.sh)
```

The LXC runs **only the serving stack** in Docker. Training (`python -m src.main train`) runs on the host with a full Python env, or on your PC with `MLFLOW_TRACKING_URI=http://<LXC-IP>:5000`.

**Do not commit** your LXC IP, `.env` file, or homelab credentials to git. Use placeholders in docs and a local `.env` for MLflow LAN access (see §5b).

---

## 1. Create the LXC (Proxmox)

Suggested starting point (adjust to your hardware):

| Setting | Suggestion |
|---------|------------|
| Template | Debian 12 or Ubuntu 22.04 / 24.04 |
| CPU | 2 cores |
| RAM | **4096 MiB** preferred (3072 MiB minimum; API + MLflow is tight on 3 GB) |
| Disk | 20 GB+ |
| Network | Bridge (vmbr0) — DHCP or static LAN IP |
| Options | **Unprivileged** + **Nesting** enabled (required for Docker in LXC) |

After creation, note the **LXC IP** privately (password manager — not in the repo).

---

## 2. Install Docker inside the LXC

SSH into the LXC:

```bash
apt update && apt upgrade -y
apt install -y git curl ca-certificates

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg \
  -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

docker --version
docker compose version
```

---

## 3. Clone the repo

```bash
cd ~
git clone https://github.com/BOYSABIO/Capstone-Microsoft.git
cd Capstone-Microsoft
git checkout main
```

---

## 4. Seed MLflow data (model store)

`mlflow.db` and `mlartifacts/` are **gitignored**. Copy them from your dev machine:

**From your PC (PowerShell, repo root):**

```powershell
scp mlflow.db <user>@<LXC-IP>:~/Capstone-Microsoft/
scp -r mlartifacts <user>@<LXC-IP>:~/Capstone-Microsoft/
```

Replace `<user>` and `<LXC-IP>` with your SSH login and the container's LAN address.

**Verify on the LXC:**

```bash
ls -la mlflow.db mlartifacts/
```

Confirm **`huckleberry-habitat`** has alias **`production`** in the MLflow UI after startup.

---

## 5. Start the stack (homelab)

On a low-RAM or slow CPU host, **do not** rely on a single `docker compose up` if the API starts before MLflow is ready. Use the helper script:

```bash
cd ~/Capstone-Microsoft
chmod +x scripts/homelab_start.sh scripts/homelab_update.sh
./scripts/homelab_start.sh
```

This starts MLflow first, waits until `http://127.0.0.1:5000/` returns **200** (can take **15–20 minutes** on first boot while `pip install mlflow` runs), then starts the API.

**Smoke tests on the LXC:**

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"year":2020,"season_num":2,"elevation":1200.0,"soil_ph":5.5,"air_temperature":15.2,"precipitation_amount":2.1,"specific_humidity":0.008,"relative_humidity":65.0,"mean_vapor_pressure_deficit":0.5,"potential_evapotranspiration":3.2,"surface_downwelling_shortwave_flux_in_air":250.0,"wind_speed":2.5}'
```

From **your PC** on the same LAN: `http://<LXC-IP>:8000/docs`

---

## 5b. MLflow UI from your PC (Invalid Host header)

MLflow 3.x blocks unknown `Host` headers. Browsing to `http://<LXC-IP>:5000` from another machine sends `Host: <LXC-IP>:5000`, which is rejected unless you allow it.

**On the LXC only** (local file, never committed):

```bash
cp .env.example .env
nano .env
```

Add your LXC's LAN address to `MLFLOW_ALLOWED_HOSTS` (see comments in `.env.example`). Example shape — **use your own IP**:

```bash
MLFLOW_ALLOWED_HOSTS=mlflow:5000,localhost:*,127.0.0.1:*,host.docker.internal:*,<YOUR-LXC-IP>:5000
```

Restart MLflow:

```bash
docker compose up -d mlflow
```

**Alternative:** SSH tunnel (no `.env` change):

```powershell
ssh -L 5000:127.0.0.1:5000 <user>@<LXC-IP>
# then open http://localhost:5000 on your PC
```

The **API on :8000** does not have this restriction — `/docs` works from the LAN without extra config.

---

## 6. Firewall / networking

If you can't reach the LXC from other machines:

1. **Proxmox host firewall** — allow TCP 8000, 5000 to the LXC.
2. **LXC `ufw`** (if enabled): `ufw allow 8000/tcp && ufw allow 5000/tcp`
3. Confirm LAN IP with `ip a`.

**Security:** Ports are exposed on your **home network only**. Do not port-forward to the internet without auth. For automated deploys, prefer Tailscale / tailnet access over exposing SSH publicly.

---

## 7. Optional: run on boot

```bash
sudo tee /etc/systemd/system/huckleberry.service << 'EOF'
[Unit]
Description=Huckleberry API + MLflow
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/YOUR_USER/Capstone-Microsoft
ExecStart=/home/YOUR_USER/Capstone-Microsoft/scripts/homelab_start.sh
ExecStop=/usr/bin/docker compose down
User=YOUR_USER

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable huckleberry.service
```

Replace `YOUR_USER` and paths.

---

## Updating the deployment

### Code changes

Develop locally → merge to `main` → on the LXC:

```bash
cd ~/Capstone-Microsoft
./scripts/homelab_update.sh
```

Or manually: `docker compose down` → `git pull origin main` → `./scripts/homelab_start.sh`

### New model (no code change)

1. Train with `MLFLOW_TRACKING_URI` pointing at homelab MLflow (or copy `mlflow.db` + `mlartifacts/`).
2. MLflow UI → set **`production`** alias.
3. `docker compose restart api`

---

## Automated CD preparation

GitHub-hosted Actions runners are not on the homelab LAN by default. For automated deploys, this setup uses **Tailscale subnet routing** to expose the services VLAN to trusted tailnet clients. The LXC host is reachable on that routed subnet, so the container itself does **not** need the Tailscale client installed.

The future deploy workflow (Issue #23) will:

1. Join the tailnet using a Tailscale auth key
2. SSH to the LXC host
3. Run deploy commands on the box

### Required GitHub Actions secrets

- `HOMELAB_HOST` — CT 103 VLAN IP / hostname (example shape: `10.x.x.13`)
- `HOMELAB_USER` — SSH login for deploys
- `HOMELAB_SSH_KEY` — private SSH deploy key stored in GitHub Actions secrets
- `TAILSCALE_AUTHKEY` — auth key for the GitHub runner to join the tailnet before SSH

### SSH key setup

A dedicated deploy key should be generated on the workstation and used only for GitHub Actions deploys. The public key belongs on the homelab box in the target user's `authorized_keys`; the private key belongs only in GitHub Actions secrets.

Example verification from the workstation:

```powershell
ssh -i $env:USERPROFILE\.ssh\huckleberry_homelab_deploy root@10.x.x.13 "whoami && hostname && pwd"
```

### Least-privilege note

For simplicity on a fresh homelab node, deploys may initially SSH as `root`. Longer-term, a dedicated deploy user is preferable. Regardless of user choice:

- use a deploy-only SSH key rather than a normal personal login key
- revoke access by removing the public key from `authorized_keys`
- rotate the GitHub secret if the key is replaced or exposed

This section prepares **credentials and connectivity only**. It does **not** add the actual deploy workflow; that belongs to Issue #23.

---

## Training on the homelab (optional)

The API image is slim. For training on the LXC:

```bash
apt install -y python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.main train --dataset hb
```

Promote in MLflow UI → `docker compose restart api`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| MLflow unhealthy / `curl :5000` → `000` for 15+ min | Normal on first boot — wait; check `docker top capstone-microsoft-mlflow-1` for `pip install` |
| API `Connection refused` to `mlflow:5000` | MLflow not ready — stop api, wait for :5000 → 200, `docker compose up -d --no-deps api` |
| `Invalid Host header` on MLflow UI from PC | Add `<YOUR-LXC-IP>:5000` to `MLFLOW_ALLOWED_HOSTS` in local `.env` (§5b) |
| `Child process died` in MLflow logs | OOM — stop api, bump LXC RAM to 4 GB; compose uses `--workers 1` |
| API exits at startup | Missing `mlflow.db` / `mlartifacts/` or no `@production` alias |
| Model stale after promote | `docker compose restart api` |
| Windows-trained artifacts on Linux | `mlflow_docker_prepare.py` runs automatically on MLflow start |

---

## Done checklist (Issue #11)

- [ ] LXC with Docker + nesting; 4 GB RAM if possible
- [ ] Repo cloned; `mlflow.db` + `mlartifacts/` copied
- [ ] `./scripts/homelab_start.sh` succeeds
- [ ] `curl http://localhost:8000/health` → `model_loaded: true`
- [ ] `http://<LXC-IP>:8000/docs` from another machine
- [ ] MLflow UI from PC (`.env` allowed-hosts or SSH tunnel)
- [ ] LXC IP stored privately (not in git)

