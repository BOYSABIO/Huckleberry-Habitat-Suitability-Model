#!/usr/bin/env python3
"""
Optional MLflow housekeeping for local dev.

Deletes experiment runs you no longer need. Does NOT touch the model registry
(huckleberry-habitat) — only experiment tracking data.

Usage:
  export MLFLOW_TRACKING_URI=http://localhost:5000

  # Preview only
  python scripts/mlflow_cleanup.py --dry-run

  # Delete junk experiments from Lesson 1 / early failed attempts
  python scripts/mlflow_cleanup.py --delete Default lesson-1-smoke

Keep: huckleberry-training
"""

from __future__ import annotations

import argparse
import os
import sys

KEEP_EXPERIMENTS = {"huckleberry-training"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean up MLflow experiments (local dev)")
    parser.add_argument(
        "--delete",
        nargs="*",
        metavar="EXPERIMENT",
        help="Experiment names to delete (e.g. Default lesson-1-smoke)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List experiments and runs without deleting",
    )
    args = parser.parse_args()

    if not os.getenv("MLFLOW_TRACKING_URI"):
        print("Set MLFLOW_TRACKING_URI (e.g. http://localhost:5000)", file=sys.stderr)
        return 1

    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    print(f"Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}\n")
    print("Experiments:")
    for exp in client.search_experiments():
        runs = client.search_runs(exp.experiment_id, max_results=500)
        keep = " (keep)" if exp.name in KEEP_EXPERIMENTS else ""
        print(f"  {exp.name}: {len(runs)} runs{keep}")

    print("\nRegistered models:")
    for model in client.search_registered_models():
        versions = client.search_model_versions(f"name='{model.name}'")
        aliases = []
        for v in versions:
            for alias in getattr(v, "aliases", []) or []:
                aliases.append(f"v{v.version}@{alias}")
        alias_str = f" aliases: {', '.join(aliases)}" if aliases else ""
        print(f"  {model.name}: {len(versions)} version(s){alias_str}")

    if args.dry_run or not args.delete:
        print("\nDry run — nothing deleted. Pass --delete Default lesson-1-smoke to remove.")
        return 0

    for name in args.delete:
        if name in KEEP_EXPERIMENTS:
            print(f"Skip {name} (protected)")
            continue
        exp = client.get_experiment_by_name(name)
        if exp is None:
            print(f"Skip {name} (not found)")
            continue
        print(f"Deleting experiment {name} ({exp.experiment_id})...")
        client.delete_experiment(exp.experiment_id)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
