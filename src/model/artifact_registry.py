"""
Artifact registry — tracks trained model versions on disk (registry.json).
"""

import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logging_config import get_logger


def serialize_training_model(model: Any) -> Dict[str, Any]:
    """Serialize a fitted training model into a standard artifact dict."""
    if hasattr(model, "to_artifact"):
        return model.to_artifact()
    return {
        "format": "huckleberry_wrapper",
        "model_type": "unknown",
        "estimator": getattr(model, "model", getattr(model, "estimator", model)),
        "scaler": getattr(model, "scaler", None),
        "feature_names": getattr(model, "feature_names", []),
        "is_fitted": getattr(model, "is_fitted", True),
    }


class ModelArtifactRegistry:
    """Versioned store for trained model artifacts."""

    def __init__(self, registry_path: str = "models/"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self.registry = self._load_registry()
        self.logger = get_logger("model_artifact_registry")

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"models": [], "current": None}

    def _save_registry(self) -> None:
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def _resolve_model_path(self, entry: Dict[str, Any]) -> Path:
        """Resolve a model file path, tolerating cross-platform path drift."""
        version_id = entry["version_id"]
        candidates = [
            Path(entry["file_path"]),
            self.registry_path / f"{version_id}.joblib",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Model file not found for {version_id}")

    def register_model(
        self,
        model,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        feature_names: List[str],
        training_data_info: Dict[str, Any],
        parameters: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_name}_v{len(self.registry['models']) + 1}_{timestamp}"
        model_file = self.registry_path / f"{version_id}.joblib"

        artifact = serialize_training_model(model)
        model_data = {
            "model": model,
            "artifact": artifact,
            "feature_names": feature_names,
            "model_type": model_type,
            "parameters": parameters,
            "version_id": version_id,
        }
        joblib.dump(model_data, model_file)

        model_entry = {
            "version_id": version_id,
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "file_path": model_file.as_posix(),
            "metrics": metrics,
            "feature_names": feature_names,
            "training_data_info": training_data_info,
            "parameters": parameters,
            "description": description,
            "tags": tags or [],
            "status": "active",
        }

        self.registry["models"].append(model_entry)
        self.registry["current"] = version_id
        self._save_registry()
        self.logger.info(f"Registered model: {version_id}")
        return version_id

    def load_model(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        if version_id is None:
            version_id = self.registry["current"]
        if version_id is None:
            raise ValueError("No current model available")

        entry = self.get_model_by_id(version_id)
        if entry is None:
            raise ValueError(f"Model version {version_id} not found")

        model_file = self._resolve_model_path(entry)
        model_data = joblib.load(model_file)
        self.logger.info(f"Loaded model: {version_id}")
        return model_data

    def get_current_model(self) -> Optional[Dict[str, Any]]:
        if self.registry["current"] is None:
            return None
        return self.get_model_by_id(self.registry["current"])

    def list_models(self) -> List[Dict[str, Any]]:
        return self.registry["models"]

    def get_model_by_id(self, version_id: str) -> Optional[Dict[str, Any]]:
        for entry in self.registry["models"]:
            if entry["version_id"] == version_id:
                return entry
        return None

    def get_latest_entry_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        matching = [e for e in self.registry["models"] if e["model_name"] == model_name]
        if not matching:
            return None
        return max(matching, key=lambda x: x["timestamp"])

    def get_latest_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        entry = self.get_latest_entry_by_name(model_name)
        if entry is None:
            return None
        model_file = self._resolve_model_path(entry)
        if not model_file.exists():
            self.logger.warning(f"Model file not found: {model_file}")
            return None
        model_data = joblib.load(model_file)
        self.logger.info(f"Loaded latest {model_name} model: {entry['version_id']}")
        return model_data

    def set_current_model(self, version_id: str) -> None:
        if self.get_model_by_id(version_id) is None:
            raise ValueError(f"Model version {version_id} not found")
        self.registry["current"] = version_id
        self._save_registry()
        self.logger.info(f"Set current model to: {version_id}")

    def delete_model(self, version_id: str) -> None:
        for i, entry in enumerate(self.registry["models"]):
            if entry["version_id"] == version_id:
                try:
                    model_file = self._resolve_model_path(entry)
                except FileNotFoundError:
                    model_file = Path(entry["file_path"])
                if model_file.exists():
                    model_file.unlink()
                del self.registry["models"][i]
                if self.registry["current"] == version_id:
                    self.registry["current"] = (
                        self.registry["models"][-1]["version_id"]
                        if self.registry["models"]
                        else None
                    )
                self._save_registry()
                self.logger.info(f"Deleted model: {version_id}")
                return
        raise ValueError(f"Model version {version_id} not found")

    def get_model_metrics(self, version_id: Optional[str] = None) -> Dict[str, float]:
        if version_id is None:
            version_id = self.registry["current"]
        if version_id is None:
            raise ValueError("No current model available")
        entry = self.get_model_by_id(version_id)
        if entry is None:
            raise ValueError(f"Model version {version_id} not found")
        return entry["metrics"]

    def compare_models(self, version_ids: List[str]) -> pd.DataFrame:
        if len(version_ids) < 2:
            raise ValueError("Need at least 2 model versions to compare")
        rows = []
        for version_id in version_ids:
            entry = self.get_model_by_id(version_id)
            if entry is None:
                raise ValueError(f"Model version {version_id} not found")
            rows.append(
                {
                    "version_id": version_id,
                    "model_name": entry["model_name"],
                    "model_type": entry["model_type"],
                    "timestamp": entry["timestamp"],
                    **entry["metrics"],
                }
            )
        return pd.DataFrame(rows)


def register_model(*args, **kwargs) -> str:
    return ModelArtifactRegistry().register_model(*args, **kwargs)


def load_current_model() -> Dict[str, Any]:
    return ModelArtifactRegistry().load_model()


# Backward-compatible alias
ModelRegistry = ModelArtifactRegistry
