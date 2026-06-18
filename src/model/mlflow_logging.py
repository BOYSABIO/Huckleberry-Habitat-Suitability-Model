"""Log training runs to MLflow (tracking + model registry)."""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc

from src.model.mlflow_config import (
    MLFLOW_EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)


class HuckleberryJoblibModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper around our existing .joblib artifact.

    At scoring time MLflow calls load_context once, then predict per request.
    The API can instead use load_predictor_from_path on the same joblib file.
    """

    def load_context(self, context) -> None:
        from src.model.predictor import load_predictor_from_path

        self.predictor = load_predictor_from_path(context.artifacts["model_file"])

    def predict(self, context, model_input: Any) -> Any:
        import numpy as np
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            features = model_input
        else:
            features = pd.DataFrame(model_input)
        return self.predictor.predict_proba(features)[:, -1]


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _log_params(params: Dict[str, Any]) -> None:
    """Log flat params; skip None and coerce values MLflow accepts."""
    for key, value in params.items():
        if value is None:
            continue
        mlflow.log_param(key, value)


def log_training_run(
    *,
    model_path: Path,
    version_id: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    feature_names: List[str],
    tags: Optional[Dict[str, str]] = None,
    register: bool = True,
) -> str:
    """
    Log params, metrics, artifacts, and register a LoggedModel for MLflow 3.

    MLflow 3 requires mlflow.*.log_model (LoggedModel) for register_model —
    log_artifact alone is not enough for the model registry.

    Returns the MLflow run_id.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"train_{version_id}") as run:
        _log_params(params)

        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        mlflow.set_tag("version_id", version_id)
        mlflow.set_tag("feature_names", ",".join(feature_names))
        commit = _git_commit()
        if commit:
            mlflow.set_tag("git_commit", commit)
        if tags:
            for key, value in tags.items():
                if value is not None:
                    mlflow.set_tag(key, str(value))

        # Human-visible copy under the run Artifacts tab
        mlflow.log_artifact(str(model_path), artifact_path="model")

        # LoggedModel for MLflow 3 registry + future model URI loading
        model_info = mlflow.pyfunc.log_model(
            python_model=HuckleberryJoblibModel(),
            name="model",
            artifacts={"model_file": str(model_path)},
        )

        if register:
            mlflow.register_model(model_info.model_uri, REGISTERED_MODEL_NAME)

        return run.info.run_id
