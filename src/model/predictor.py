"""
Load saved models and score prepared feature matrices.

Lives under model/ (not inference/) because loading and predict_proba are core
model concerns. The inference package orchestrates coordinates → features → predictor.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config.settings import Settings
from src.model.implementations.ensemble import EnsembleModel
from src.model.implementations.random_forest import RandomForestModel
from src.model.store import ModelArtifactRegistry


@dataclass
class PredictionResult:
    """Result of a prediction."""
    predictions: np.ndarray
    probabilities: np.ndarray
    confidence_intervals: Optional[List[Tuple[float, float]]] = None


class HabitatPredictor:
    """Score habitat suitability from a prepared feature matrix (features in → scores out)."""

    def __init__(
        self,
        feature_names: List[str],
        version_id: str = "unknown",
        wrapper: Optional[Any] = None,
        estimator: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ):
        self.feature_names = feature_names
        self.version_id = version_id
        self._wrapper = wrapper
        self._estimator = estimator
        self._scaler = scaler

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return features[self.feature_names].fillna(0)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the class of a sample."""
        prepared = self._prepare_features(features)
        if self._wrapper is not None:
            return self._wrapper.predict(prepared)
        if self._scaler is not None:
            return self._estimator.predict(self._scaler.transform(prepared))
        return self._estimator.predict(prepared)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the probability of a sample being in the positive class."""
        prepared = self._prepare_features(features)
        if self._wrapper is not None:
            return self._wrapper.predict_proba(prepared)
        if self._scaler is not None:
            return self._estimator.predict_proba(self._scaler.transform(prepared))
        return self._estimator.predict_proba(prepared)

    def predict_with_interval(
        self,
        features: pd.DataFrame,
        percentile_range: Tuple[float, float] = (2.5, 97.5),
    ) -> PredictionResult:
        """Predict the class of a sample with a confidence interval."""
        prepared = self._prepare_features(features)
        probabilities = self.predict_proba(prepared)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        intervals = None
        rf_estimator = self._resolve_random_forest_estimator()
        if rf_estimator is not None:
            intervals = self._tree_confidence_intervals(
                rf_estimator, prepared, percentile_range
            )

        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence_intervals=intervals,
        )

    def _resolve_random_forest_estimator(self) -> Optional[RandomForestClassifier]:
        if isinstance(self._wrapper, RandomForestModel):
            return self._wrapper.estimator
        if isinstance(self._estimator, RandomForestClassifier):
            return self._estimator
        return None

    def _tree_confidence_intervals(
        self,
        estimator: RandomForestClassifier,
        features: pd.DataFrame,
        percentile_range: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        if self._scaler is not None and self._wrapper is None:
            matrix = self._scaler.transform(features)
        elif isinstance(self._wrapper, RandomForestModel):
            matrix = self._wrapper._prepare_scaled(features)
        else:
            matrix = features.values

        intervals: List[Tuple[float, float]] = []
        for row in matrix:
            tree_probs = np.array(
                [tree.predict_proba(row.reshape(1, -1))[0, 1] for tree in estimator.estimators_]
            )
            low, high = np.percentile(tree_probs, percentile_range)
            intervals.append((float(low), float(high)))
        return intervals


def _load_joblib_payload(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _predictor_from_payload(payload: Any, version_id: str, source_path: Path) -> HabitatPredictor:
    if isinstance(payload, (RandomForestModel, EnsembleModel)):
        return HabitatPredictor(
            feature_names=payload.feature_names,
            version_id=version_id,
            wrapper=payload,
        )

    if isinstance(payload, dict) and payload.get("format") == "huckleberry_wrapper":
        model_type = payload.get("model_type", "random_forest")
        wrapper = (
            EnsembleModel.from_artifact(payload)
            if model_type == "ensemble"
            else RandomForestModel.from_artifact(payload)
        )
        return HabitatPredictor(
            feature_names=wrapper.feature_names,
            version_id=version_id,
            wrapper=wrapper,
        )

    if isinstance(payload, dict) and "model" in payload:
        inner = payload["model"]
        feature_names = payload.get("feature_names") or getattr(
            inner, "feature_names", None
        ) or list(getattr(inner, "feature_names_in_", []))

        if isinstance(inner, (RandomForestModel, EnsembleModel)):
            return HabitatPredictor(
                feature_names=inner.feature_names,
                version_id=version_id,
                wrapper=inner,
            )

        if isinstance(inner, dict) and "estimator" in inner:
            return HabitatPredictor(
                feature_names=inner["feature_names"],
                version_id=version_id,
                estimator=inner["estimator"],
                scaler=inner.get("scaler"),
            )

        if hasattr(inner, "predict_proba") and hasattr(inner, "feature_names"):
            return HabitatPredictor(
                feature_names=list(inner.feature_names),
                version_id=version_id,
                wrapper=inner,
            )

        scaler = payload.get("scaler")
        if scaler is not None:
            return HabitatPredictor(
                feature_names=list(feature_names),
                version_id=version_id,
                estimator=inner,
                scaler=scaler,
            )

        if hasattr(inner, "feature_names_in_"):
            return HabitatPredictor(
                feature_names=list(inner.feature_names_in_),
                version_id=version_id,
                estimator=inner,
            )

    if hasattr(payload, "feature_names_in_"):
        return HabitatPredictor(
            feature_names=list(payload.feature_names_in_),
            version_id=version_id,
            estimator=payload,
        )

    raise ValueError(f"Unsupported model format in {source_path}")


def load_predictor_from_path(path: Union[str, Path]) -> HabitatPredictor:
    """Load model from path."""
    source_path = Path(path)
    payload = _load_joblib_payload(source_path)
    return _predictor_from_payload(payload, version_id=source_path.stem, source_path=source_path)


def load_predictor_from_settings(settings: Settings) -> HabitatPredictor:
    """Load model from settings."""
    if settings.inference.model_file_path:
        return load_predictor_from_path(settings.inference.model_file_path)

    store = ModelArtifactRegistry(settings.model.model_registry_path)
    version_id = store.registry.get("current", "registry_current")
    payload = store.load_model()
    entry = store.get_model_by_id(version_id) if version_id else None
    source = store._resolve_model_path(entry) if entry else Path("registry")

    inner = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return _predictor_from_payload(inner, version_id=version_id, source_path=source)


def load_predictor_for_api(
    model_path: Optional[str] = None,
    registry_path: str = "models/"
) -> HabitatPredictor:
    """
    Load model from path or registry.

    MODEL_PATH takes priority over registry_path.
    """

    if model_path:
        return load_predictor_from_path(model_path)

    store = ModelArtifactRegistry(registry_path)
    version_id = store.registry.get("current")
    if not version_id:
        raise FileNotFoundError(
            "No model registered. Train one with:\n"
            "python -m src.main train --dataset hb\n"
            "Or set MODEL_PATH to a .joblib file"
        )

    entry = store.get_model_by_id(version_id)
    if entry is None:
        raise FileNotFoundError(f"Model version {version_id} not found")
    try:
        source = store._resolve_model_path(entry)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Registry points to '{version_id}' but file not found\n"
            f"  {exc}\n"
            "Train a new model with:\n"
            "  python -m src.main train --dataset hb\n"
            "Or set MODEL_PATH to a .joblib file"
        )
    return load_predictor_from_path(source)
