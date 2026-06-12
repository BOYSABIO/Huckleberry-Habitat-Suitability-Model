"""
Location-based inference orchestration.

pipeline.py coordinates the full infer workflow:
  coordinates → environmental features → predictor → reports

For feature-only scoring (API path), use HabitatPredictor in model.predictor directly.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config.settings import Settings
from src.data_validation.validate import validate_inference_data
from src.features.environmental import EnvironmentalDataExtractor
from src.features.temporal import add_season_column
from src.inference import reporting
from src.model.predictor import HabitatPredictor, load_predictor_from_settings
from src.utils.logging_config import get_logger, log_pipeline_step

logger = get_logger("inference_pipeline")


class InferencePipeline:
    """Orchestrate coordinate-based habitat suitability inference."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.env_extractor = EnvironmentalDataExtractor()
        self.predictor: HabitatPredictor = load_predictor_from_settings(self.settings)
        logger.info(f"Loaded model: {self.predictor.version_id}")

    @log_pipeline_step("Input Validation")
    def validate_input(self, coordinates: List[Tuple[float, float]]) -> pd.DataFrame:
        df = pd.DataFrame(coordinates, columns=["decimalLatitude", "decimalLongitude"])
        if not validate_inference_data(df, self.settings.inference.required_columns):
            raise ValueError("Input validation failed")
        return df

    @log_pipeline_step("Environmental Data Extraction")
    def extract_environmental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        target_date = None
        if not self.settings.inference.use_latest_gridmet and self.settings.inference.gridmet_date:
            target_date = self.settings.inference.gridmet_date

        df = self.env_extractor.extract_gridmet_data(df, target_date=target_date)
        if len(df) == 0:
            raise ValueError("No coordinates within GridMET bounds")

        df = self.env_extractor.add_elevation_data(df)
        df = self.env_extractor.add_soil_data(df)

        if "datetime" in df.columns:
            df["year"] = df["datetime"].dt.year
            df["month"] = df["datetime"].dt.month
            df["day"] = df["datetime"].dt.day
            df = add_season_column(df)

        return df

    @log_pipeline_step("Feature Preparation")
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.predictor.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return df[self.predictor.feature_names].fillna(0)

    @log_pipeline_step("Model Prediction")
    def make_predictions(self, feature_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predictor.predict_proba(feature_df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        results_df = original_df.copy()
        results_df["prediction"] = predictions
        results_df["probability"] = probabilities
        return results_df

    def run(
        self,
        coordinates: List[Tuple[float, float]],
        create_map: bool = True,
        confidence_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        logger.info("Starting inference pipeline")
        df = self.validate_input(coordinates)
        df = self.extract_environmental_data(df)
        feature_df = self.prepare_features(df)
        results_df = self.make_predictions(feature_df, df)

        csv_path = reporting.save_predictions_csv(results_df)
        summary_path = reporting.generate_inference_summary(
            results_df,
            self.predictor.version_id,
            confidence_threshold,
        )
        top_predictions_path = reporting.save_top_predictions(results_df, confidence_threshold)
        confidence_plot_path = reporting.create_confidence_plot(results_df)

        map_path = None
        if create_map:
            map_path = reporting.create_prediction_map(
                results_df,
                confidence_threshold=confidence_threshold,
            )

        suitable_count = int((results_df["probability"] >= confidence_threshold).sum())
        return {
            "success": True,
            "total_coordinates": len(coordinates),
            "valid_coordinates": len(results_df),
            "suitable_habitat_count": suitable_count,
            "average_confidence": float(results_df["probability"].mean()),
            "predictions": results_df,
            "csv_path": csv_path,
            "map_path": map_path,
            "summary_path": summary_path,
            "top_predictions_path": top_predictions_path,
            "confidence_plot_path": confidence_plot_path,
            "model_version": self.predictor.version_id,
        }
