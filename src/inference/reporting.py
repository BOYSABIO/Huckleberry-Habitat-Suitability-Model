"""
Inference output generation — maps, CSVs, summaries, and plots.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger("inference_reporting")


def save_predictions_csv(
    results_df: pd.DataFrame,
    output_path: str = "outputs/predictions/inference_predictions.csv",
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)
    logger.info(f"Predictions saved to: {output}")
    return str(output)


def generate_inference_summary(
    results_df: pd.DataFrame,
    model_version: str,
    confidence_threshold: float,
) -> Optional[str]:
    try:
        total = len(results_df)
        suitable = int((results_df["probability"] >= confidence_threshold).sum())
        summary = {
            "inference_summary": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_version": model_version,
                "total_coordinates": total,
                "valid_coordinates": total,
                "suitable_habitat_count": suitable,
                "suitability_percentage": (suitable / total) * 100 if total else 0,
                "average_confidence": round(float(results_df["probability"].mean()), 4),
                "confidence_threshold": confidence_threshold,
                "confidence_distribution": {
                    "high_confidence": int((results_df["probability"] >= 0.8).sum()),
                    "medium_confidence": int(
                        ((results_df["probability"] >= 0.6) & (results_df["probability"] < 0.8)).sum()
                    ),
                    "low_confidence": int((results_df["probability"] < 0.6).sum()),
                },
                "top_locations": results_df.nlargest(5, "probability")[
                    ["decimalLatitude", "decimalLongitude", "probability"]
                ].to_dict("records"),
            }
        }
        output_dir = Path("outputs/summaries")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"inference_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Inference summary saved: {summary_path}")
        return str(summary_path)
    except Exception as exc:
        logger.error(f"Summary generation failed: {exc}")
        return None


def save_top_predictions(
    results_df: pd.DataFrame,
    confidence_threshold: float,
) -> Optional[str]:
    top_predictions = results_df[results_df["probability"] >= confidence_threshold].copy()
    if len(top_predictions) == 0:
        logger.warning("No high-confidence predictions found")
        return None
    top_predictions = top_predictions.sort_values("probability", ascending=False)
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"top_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    top_predictions.to_csv(path, index=False)
    logger.info(f"Top predictions saved: {path}")
    return str(path)


def create_prediction_map(
    results_df: pd.DataFrame,
    output_path: str = "outputs/maps/prediction_map.html",
    confidence_threshold: float = 0.8,
) -> Optional[str]:
    import folium

    suitable_df = results_df[results_df["probability"] >= confidence_threshold].copy()
    if len(suitable_df) == 0:
        logger.warning("No suitable habitat found above confidence threshold")
        return None

    map_center = [
        suitable_df["decimalLatitude"].mean(),
        suitable_df["decimalLongitude"].mean(),
    ]
    m = folium.Map(location=map_center, zoom_start=8)
    for _, row in suitable_df.iterrows():
        popup_text = (
            f"<b>Suitable Huckleberry Habitat</b><br>"
            f"Confidence: {row['probability']:.2%}<br>"
            f"Latitude: {row['decimalLatitude']:.4f}<br>"
            f"Longitude: {row['decimalLongitude']:.4f}<br>"
        )
        folium.Marker(
            location=[row["decimalLatitude"], row["decimalLongitude"]],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="green", icon="leaf"),
        ).add_to(m)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    logger.info(f"Prediction map saved to: {output}")
    return str(output)


def create_confidence_plot(results_df: pd.DataFrame) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        ax1.hist(results_df["probability"], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_title("Confidence Distribution")

        colors = ["red" if p < 0.5 else "green" for p in results_df["probability"]]
        ax2.scatter(range(len(results_df)), results_df["probability"], c=colors, alpha=0.6)
        ax2.axhline(y=0.5, color="orange", linestyle="--")
        ax2.set_title("Confidence by Location")

        high = int((results_df["probability"] >= 0.8).sum())
        med = int(((results_df["probability"] >= 0.6) & (results_df["probability"] < 0.8)).sum())
        low = int((results_df["probability"] < 0.6).sum())
        ax3.pie(
            [high, med, low],
            labels=["High (≥0.8)", "Medium (0.6-0.8)", "Low (<0.6)"],
            colors=["green", "orange", "red"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax3.set_title("Confidence Categories")

        ax4.axis("off")
        ax4.set_title("Summary Statistics")
        plt.tight_layout()

        output_dir = Path("outputs/summaries")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"confidence_plot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Confidence plot saved: {plot_path}")
        return str(plot_path)
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plot")
        return None
    except Exception as exc:
        logger.error(f"Plot creation failed: {exc}")
        return None
