"""
Rewrite Windows host artifact paths for MLflow in Docker.

When training on Windows against a local mlflow server, artifact_uri values
are stored as file:///C:/Users/.../mlartifacts/... which Linux containers
cannot read. Run this before `mlflow server` in docker-compose (see compose file).

Safe to run repeatedly — only replaces known host prefixes with /data/mlartifacts.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def _docker_artifact_uri(windows_uri: str, docker_root: str) -> str | None:
    """Map file:///C:/.../mlartifacts/... to file:///data/mlartifacts/..."""
    marker = "/mlartifacts/"
    if not windows_uri or marker not in windows_uri:
        return None
    suffix = windows_uri.split(marker, 1)[1]
    root = docker_root.rstrip("/")
    return f"file:///{root.lstrip('/')}/{suffix}"


def fix_mlflow_db(db_path: Path, docker_root: str) -> int:
    """Replace file:// artifact roots in mlflow.db. Returns number of row updates."""
    if not db_path.is_file():
        return 0

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    updates = 0

    cur.execute("SELECT run_uuid, artifact_uri FROM runs WHERE artifact_uri IS NOT NULL")
    for run_uuid, artifact_uri in cur.fetchall():
        new_uri = _docker_artifact_uri(artifact_uri, docker_root)
        if new_uri and new_uri != artifact_uri:
            cur.execute(
                "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
                (new_uri, run_uuid),
            )
            updates += 1

    cur.execute(
        "SELECT name, version, storage_location FROM model_versions "
        "WHERE storage_location IS NOT NULL"
    )
    for name, version, storage_location in cur.fetchall():
        new_loc = _docker_artifact_uri(storage_location, docker_root)
        if new_loc and new_loc != storage_location:
            cur.execute(
                "UPDATE model_versions SET storage_location = ? WHERE name = ? AND version = ?",
                (new_loc, name, version),
            )
            updates += 1

    con.commit()
    con.close()
    return updates


def fix_mlmodel_files(artifacts_dir: Path, docker_root: str) -> int:
    """Patch artifact_path lines inside MLmodel YAML files on disk."""
    if not artifacts_dir.is_dir():
        return 0

    count = 0
    for mlmodel in artifacts_dir.rglob("MLmodel"):
        text = mlmodel.read_text(encoding="utf-8")
        if "artifact_path:" not in text:
            continue
        lines = []
        changed = False
        for line in text.splitlines():
            if line.startswith("artifact_path:") and "/mlartifacts/" in line:
                raw = line.split(":", 1)[1].strip()
                new_path = _docker_artifact_uri(raw, docker_root)
                if new_path and new_path != raw:
                    line = f"artifact_path: {new_path}"
                    changed = True
            lines.append(line)
        if changed:
            mlmodel.write_text("\n".join(lines) + "\n", encoding="utf-8")
            count += 1
    return count


def main() -> None:
    data_dir = Path(os.environ.get("MLFLOW_DATA_DIR", "/data"))
    db_path = data_dir / "mlflow.db"
    artifacts_dir = data_dir / "mlartifacts"
    docker_root = os.environ.get("MLFLOW_DOCKER_ARTIFACT_ROOT", "/data/mlartifacts")

    db_updates = fix_mlflow_db(db_path, docker_root)
    file_updates = fix_mlmodel_files(artifacts_dir, docker_root)
    print(
        f"mlflow_docker_prepare: db_rows={db_updates}, mlmodel_files={file_updates}, "
        f"root={docker_root}"
    )


if __name__ == "__main__":
    main()
