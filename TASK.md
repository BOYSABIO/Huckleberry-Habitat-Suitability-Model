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
