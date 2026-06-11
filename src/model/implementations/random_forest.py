"""
Random Forest model implementation.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.model.base import prepare_training_features, select_inference_features


class RandomForestModel:
    """Random Forest classifier with feature scaling."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: Optional[list] = None
        self.is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = "occurrence",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        features, target, feature_names = prepare_training_features(data, target_col)
        self.feature_names = feature_names

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
        )

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        self.estimator.fit(x_train_scaled, y_train)

        y_pred = self.estimator.predict(x_test_scaled)
        self.is_fitted = True

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "test_size": len(x_test),
            "train_size": len(x_train),
        }

    def _prepare_scaled(self, data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        prepared = select_inference_features(data, self.feature_names)
        return self.scaler.transform(prepared)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._prepare_scaled(data))

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self._prepare_scaled(data))

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.estimator.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def to_artifact(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before serialization")
        return {
            "format": "huckleberry_wrapper",
            "model_type": "random_forest",
            "estimator": self.estimator,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "RandomForestModel":
        instance = cls()
        instance.estimator = artifact["estimator"]
        instance.scaler = artifact["scaler"]
        instance.feature_names = artifact["feature_names"]
        instance.is_fitted = artifact.get("is_fitted", True)
        return instance
