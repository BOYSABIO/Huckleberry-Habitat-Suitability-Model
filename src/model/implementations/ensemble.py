"""
TPOT stacking ensemble model implementation.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model.features import prepare_training_features, select_inference_features
from src.model.registry import register


@register("ensemble")
class EnsembleModel:
    """XGBoost + Bernoulli Naive Bayes stacking ensemble from TPOT experiments."""

    def __init__(self):
        self.estimator = None
        self.scaler = StandardScaler()
        self.feature_names: Optional[list] = None
        self.is_fitted = False

    def _create_pipeline(self) -> Any:
        from tpot.builtins import StackingEstimator
        from tpot.export_utils import set_param_recursive
        from xgboost import XGBClassifier

        pipeline = make_pipeline(
            StackingEstimator(
                estimator=XGBClassifier(
                    learning_rate=0.5,
                    max_depth=3,
                    min_child_weight=8,
                    n_estimators=100,
                    n_jobs=1,
                    subsample=1.0,
                    verbosity=0,
                )
            ),
            BernoulliNB(alpha=10.0, fit_prior=False),
        )
        set_param_recursive(pipeline.steps, "random_state", 42)
        return pipeline

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

        self.estimator = self._create_pipeline()
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

        try:
            stacking_estimator = self.estimator.named_steps["stackingestimator"]
            importance = stacking_estimator.estimator.feature_importances_
        except (KeyError, AttributeError):
            return pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": [0.0] * len(self.feature_names),
                }
            ).sort_values("importance", ascending=False)

        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def to_artifact(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before serialization")
        return {
            "format": "huckleberry_wrapper",
            "model_type": "ensemble",
            "estimator": self.estimator,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "EnsembleModel":
        instance = cls()
        instance.estimator = artifact["estimator"]
        instance.scaler = artifact["scaler"]
        instance.feature_names = artifact["feature_names"]
        instance.is_fitted = artifact.get("is_fitted", True)
        return instance
