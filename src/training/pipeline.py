"""Training pipeline for the Huckleberry Habitat Prediction Pipeline."""

import os
from typing import Any, Dict, Optional

import pandas as pd

from src.config.environments import ENRICHED_OUTPUT, PROCESSED_OUTPUT
from src.config.settings import Settings
from src.data_load import DataLoader
from src.data_preprocess.geocode import Geocoder, apply_manual_geocodes, load_manual_geocodes
from src.data_preprocess.preprocessor import DataPreprocessor
from src.data_validation.validate import validate_data
from src.evaluation.feature_importance import generate_feature_importance_outputs
from src.features.environmental import EnvironmentalDataExtractor
from src.features.sampling import create_pseudo_absences
from src.model.store import ModelArtifactRegistry
from src.model.trainer import train_model as run_model_training
from src.utils.data_versioning import DataVersioning
from src.utils.logging_config import get_logger, log_pipeline_step


class TrainingPipeline:
    """Training pipeline orchestrator."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.logger = get_logger("training_pipeline")
        self.data_versioning = DataVersioning()
        self.model_registry = ModelArtifactRegistry(self.settings.model.model_registry_path)

        self.data_loader = DataLoader(self.settings.data)
        self.preprocessor = DataPreprocessor()
        self.geocoder = Geocoder()
        self.env_extractor = EnvironmentalDataExtractor()

    def _uses_training_dataset(self) -> bool:
        return bool(self.settings.data.training_dataset_path)

    @log_pipeline_step("Data Loading")
    def load_data(self) -> pd.DataFrame:
        if self._uses_training_dataset():
            path = self.settings.data.training_dataset_path
            self.logger.info("Loading pre-enriched training dataset: %s", path)
            return self.data_loader.load_training_dataset(path)

        self.logger.info("Loading raw data from: %s", self.settings.data.raw_data_path)
        df = self.data_loader.load_gbif_occurrences(self.settings.data.raw_data_path)
        self.logger.info("Loaded %s raw records", len(df))
        return df

    def _prepare_snapshot_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a pre-enriched CSV for validation and training."""
        df = df.copy()
        if "occurrence" not in df.columns:
            raise ValueError("Training dataset must include an 'occurrence' column")
        if "datetime" not in df.columns and {"year", "month", "day"}.issubset(df.columns):
            df["datetime"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
        if "season_num" not in df.columns and "month" in df.columns:
            from src.features.temporal import add_season_column
            df = add_season_column(df)
        return df

    @log_pipeline_step("Data Preprocessing")
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocessor.clean_occurrence_data(df)
        self.logger.info("After cleaning: %s records", len(df))
        df = self.preprocessor.filter_gridmet_time_bounds(df)
        self.logger.info("After temporal filtering: %s records", len(df))
        return df

    @log_pipeline_step("Geocoding")
    def geocode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.geocoder.geocode_dataset(df)
        self.logger.info("After geocoding: %s records", len(df))
        df = self.preprocessor.filter_gridmet_bounds(df)
        self.logger.info("After spatial filtering: %s records", len(df))
        manual_dict = load_manual_geocodes()
        df = apply_manual_geocodes(df, manual_dict)
        self.logger.info("After manual geocode fallback: %s records", len(df))
        return df

    @log_pipeline_step("Pseudo-absence Generation")
    def generate_pseudo_absences(self, df: pd.DataFrame) -> pd.DataFrame:
        combined_df = create_pseudo_absences(
            df,
            ratio=self.settings.data.pseudo_absence_ratio,
            buffer_km=self.settings.data.pseudo_absence_buffer_km,
            random_seed=self.settings.data.random_seed,
        )
        self.logger.info("Generated %s pseudo-absences", sum(combined_df["occurrence"] == 0))
        self.logger.info("Total records: %s", len(combined_df))
        return combined_df

    @log_pipeline_step("Environmental Data Extraction")
    def extract_environmental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_gridmet = self.env_extractor.extract_gridmet_data(df)
        self.logger.info("Records with GridMET data: %s/%s", len(df_with_gridmet), len(df))
        if len(df_with_gridmet) == 0:
            raise ValueError("No records within GridMET bounds. Cannot proceed.")
        return df_with_gridmet

    @log_pipeline_step("Feature Engineering")
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.env_extractor.add_elevation_data(df)
        df = self.env_extractor.add_soil_data(df)
        self.logger.info("Feature engineering complete: %s", df.shape)
        return df

    @log_pipeline_step("Data Validation")
    def validate_data(self, df: pd.DataFrame) -> bool:
        expected_columns = [
            "decimalLatitude",
            "decimalLongitude",
            "year",
            "month",
            "day",
            "datetime",
            "occurrence",
        ]
        if not validate_data(df, expected_columns):
            raise ValueError("Data validation failed")
        return True

    @log_pipeline_step("Model Training")
    def train_model(self, df: pd.DataFrame):
        self.logger.info("Training %s model", self.settings.model.model_type)
        model, metrics = run_model_training(df, self.settings.model)
        self.logger.info("Training completed. Metrics: %s", metrics)
        return model, metrics

    @log_pipeline_step("Model Registration")
    def register_model(self, model, metrics: Dict[str, float], df: pd.DataFrame) -> str:
        training_data_info = {
            "total_records": len(df),
            "occurrence_distribution": df[self.settings.model.target_column].value_counts().to_dict(),
            "feature_count": len(model.feature_names) if hasattr(model, "feature_names") else None,
        }
        version_id = self.model_registry.register_model(
            model=model,
            model_name=self.settings.model.model_name,
            model_type=self.settings.model.model_type,
            metrics=metrics,
            feature_names=model.feature_names if hasattr(model, "feature_names") else [],
            training_data_info=training_data_info,
            parameters={
                "n_estimators": self.settings.model.n_estimators,
                "test_size": self.settings.model.test_size,
                "random_state": self.settings.model.random_state,
            },
            description=f"Trained {self.settings.model.model_type} model",
            tags=["huckleberry", "habitat_prediction", self.settings.model.model_type],
        )
        self.logger.info("Model registered with version ID: %s", version_id)
        return version_id

    @log_pipeline_step("Column Selection")
    def select_modeling_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        essential_cols = [
            "decimalLatitude",
            "decimalLongitude",
            "year",
            "month",
            "day",
            "datetime",
            "occurrence",
            "season_num",
        ]
        environmental_cols = [
            "elevation",
            "soil_ph",
            "gridmet_lat",
            "gridmet_lon",
            "gridmet_date",
        ]
        gridmet_cols = [
            col
            for col in df.columns
            if col
            in {
                "air_temperature",
                "precipitation_amount",
                "specific_humidity",
                "relative_humidity",
                "mean_vapor_pressure_deficit",
                "potential_evapotranspiration",
                "surface_downwelling_shortwave_flux_in_air",
                "wind_speed",
            }
        ]
        available_cols = [col for col in essential_cols + environmental_cols + gridmet_cols if col in df.columns]
        if "occurrence" not in available_cols and "occurrence" in df.columns:
            available_cols.append("occurrence")
        return df[available_cols].copy()

    @log_pipeline_step("Essential Column Selection")
    def select_essential_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        essential_cols = [
            "gbifID",
            "decimalLatitude",
            "decimalLongitude",
            "year",
            "month",
            "day",
            "datetime",
            "occurrence",
            "season_num",
        ]
        available_cols = [col for col in essential_cols if col in df.columns]
        return df[available_cols].copy()

    @log_pipeline_step("Feature Importance Generation")
    def generate_feature_importance_outputs(self, model, version_id: str) -> None:
        generate_feature_importance_outputs(model, version_id)

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting training pipeline")
        try:
            from_snapshot = self._uses_training_dataset()
            input_source = (
                self.settings.data.training_dataset_path
                if from_snapshot
                else self.settings.data.raw_data_path
            )

            df = self.load_data()

            if from_snapshot:
                df = self._prepare_snapshot_dataset(df)
                self.logger.info("Skipping ETL — training from pre-enriched dataset")
            else:
                df = self.preprocess_data(df)
                df = self.geocode_data(df)
                df = self.generate_pseudo_absences(df)
                df = self.extract_environmental_data(df)
                df = self.engineer_features(df)

            self.validate_data(df)

            if not from_snapshot:
                df_essential = self.select_essential_columns(df)
                self.data_loader.save_processed_data(df_essential, PROCESSED_OUTPUT)
                self.logger.info(
                    "Processed data saved to: %s",
                    self.settings.get_processed_data_path(PROCESSED_OUTPUT),
                )

            df = self.select_modeling_columns(df)
            self.data_loader.save_enriched_data(df, ENRICHED_OUTPUT)
            self.logger.info(
                "Enriched data saved to: %s",
                self.settings.get_enriched_data_path(ENRICHED_OUTPUT),
            )

            model, metrics = self.train_model(df)
            version_id = self.register_model(model, metrics, df)

            if os.getenv("MLFLOW_TRACKING_URI"):
                from src.model.mlflow_logging import log_training_run

                entry = self.model_registry.get_model_by_id(version_id)
                model_path = self.model_registry._resolve_model_path(entry)
                run_id = log_training_run(
                    model_path=model_path,
                    version_id=version_id,
                    metrics=metrics,
                    params={
                        "model_type": self.settings.model.model_type,
                        "n_estimators": self.settings.model.n_estimators,
                        "test_size": self.settings.model.test_size,
                        "random_state": self.settings.model.random_state,
                        "dataset": input_source,
                        "pseudo_absence_ratio": self.settings.data.pseudo_absence_ratio,
                        "pseudo_absence_buffer_km": self.settings.data.pseudo_absence_buffer_km,
                    },
                    feature_names=model.feature_names,
                    tags={"dataset_path": input_source},
                )
                self.logger.info("MLflow run logged: %s", run_id)
            else:
                self.logger.info(
                    "MLFLOW_TRACKING_URI not set — skipping MLflow logging "
                    "(local registry still updated)"
                )

            self.generate_feature_importance_outputs(model, version_id)

            version_id_data = self.data_versioning.track_transformation(
                df=df,
                description="Training from snapshot" if from_snapshot else "Complete training pipeline run",
                input_files=[input_source],
                output_files=[PROCESSED_OUTPUT, ENRICHED_OUTPUT],
                parameters={
                    "pseudo_absence_ratio": self.settings.data.pseudo_absence_ratio,
                    "pseudo_absence_buffer_km": self.settings.data.pseudo_absence_buffer_km,
                    "model_type": self.settings.model.model_type,
                    "n_estimators": self.settings.model.n_estimators,
                },
                metadata={
                    "model_version_id": version_id,
                    "training_metrics": metrics,
                },
            )

            return {
                "success": True,
                "model_version_id": version_id,
                "data_version_id": version_id_data,
                "metrics": metrics,
                "processed_data_path": PROCESSED_OUTPUT,
                "enriched_data_path": ENRICHED_OUTPUT,
                "final_record_count": len(df),
            }
        except Exception as e:
            self.logger.error("Training pipeline failed: %s", e)
            raise
