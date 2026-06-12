"""
Feature importance reporting for trained models.
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger("feature_importance")


def extract_feature_importance(model: Any) -> Optional[pd.DataFrame]:
    if not hasattr(model, "get_feature_importance"):
        logger.warning("Model does not support feature importance")
        return None
    importance_df = model.get_feature_importance()
    if importance_df is None or importance_df.empty:
        logger.error("Feature importance DataFrame is empty")
        return None
    return importance_df


def save_training_outputs(importance_df: pd.DataFrame, version_id: str) -> None:
    output_dir = Path("outputs/feature_importance")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"feature_importance_v{version_id}.csv"
    importance_df.to_csv(csv_path, index=False)
    logger.info(f"Feature importance CSV saved: {csv_path}")

    _create_feature_importance_plot(importance_df, version_id)


def _create_feature_importance_plot(importance_df: pd.DataFrame, version_id: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        top_features = importance_df.head(10)
        colors = sns.color_palette("husl", len(top_features))
        ax1.barh(range(len(top_features)), top_features["importance"], color=colors)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features["feature"])
        ax1.set_title(f"Top 10 Feature Importance (Model v{version_id})")
        ax1.invert_yaxis()

        top_5 = importance_df.head(5)
        other = importance_df.iloc[5:]["importance"].sum()
        ax2.pie(
            list(top_5["importance"]) + [other],
            labels=list(top_5["feature"]) + ["Others"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title(f"Feature Importance Distribution (Model v{version_id})")
        plt.tight_layout()

        output_dir = Path("outputs/feature_importance")
        plot_path = output_dir / f"feature_importance_plot_v{version_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Feature importance plot saved: {plot_path}")
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plot")
    except Exception as exc:
        logger.error(f"Plot creation failed: {exc}")


def generate_feature_importance_outputs(model: Any, version_id: str) -> None:
    importance_df = extract_feature_importance(model)
    if importance_df is not None:
        save_training_outputs(importance_df, version_id)
