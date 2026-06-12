"""
Pipeline configuration presets.

Use ``get_settings(sample=False)`` for the full dataset (default).
Use ``get_settings(sample=True)`` for a quick run on the small sample file.
Use ``training_dataset=...`` to train from a pre-enriched CSV (skips ETL).
"""

from typing import Optional

from .settings import Settings, DataSettings, ModelSettings, InferenceSettings, LoggingSettings

PROCESSED_OUTPUT = "huckleberry_processed.csv"
ENRICHED_OUTPUT = "huckleberry_enriched.csv"

DEFAULT_MODEL = "models/huckleberry_model_v13_20260612_111519.joblib"

# Short names for --dataset (expand to paths)
DATASET_PRESETS = {
    "hb": "data/snapshots/HB.csv",
    "hb_full": "data/snapshots/HB_PSEUDO_clean_elevation_soil.csv",
}


def resolve_dataset_path(dataset: Optional[str]) -> Optional[str]:
    """Resolve a preset name or file path to a concrete CSV path."""
    if dataset is None:
        return None
    return DATASET_PRESETS.get(dataset, dataset)


def get_settings(
    sample: Optional[bool] = None,
    training_dataset: Optional[str] = None,
    model_type: Optional[str] = None,
) -> Settings:
    """
    Return pipeline settings.

    Args:
        sample: Use the small GBIF sample with full ETL (mutually exclusive with training_dataset).
        training_dataset: Preset name (e.g. ``hb``) or path to a pre-enriched CSV; skips ETL.
        model_type: Registered model type name (e.g. ``random_forest``, ``ensemble``).
    """
    if sample is None:
        sample = False

    dataset_path = resolve_dataset_path(training_dataset)
    if sample and dataset_path:
        raise ValueError("Use either --sample or --dataset, not both")

    model_kwargs = {}
    if model_type is not None:
        model_kwargs["model_type"] = model_type

    if dataset_path:
        return Settings(
            data=DataSettings(training_dataset_path=dataset_path),
            model=ModelSettings(n_estimators=200, model_name="huckleberry_model", **model_kwargs),
            inference=InferenceSettings(model_file_path=DEFAULT_MODEL),
            logging=LoggingSettings(level="INFO", log_file="logs/pipeline.log"),
        )

    if sample:
        return Settings(
            data=DataSettings(
                raw_data_path="data/raw/occurrence_sample.txt",
                pseudo_absence_ratio=2,
                pseudo_absence_buffer_km=2.0,
            ),
            model=ModelSettings(n_estimators=50, model_name="huckleberry_model", **model_kwargs),
            inference=InferenceSettings(model_file_path=DEFAULT_MODEL),
            logging=LoggingSettings(level="DEBUG", log_file="logs/pipeline.log"),
        )

    return Settings(
        data=DataSettings(raw_data_path="data/raw/occurrence.txt"),
        model=ModelSettings(n_estimators=200, model_name="huckleberry_model", **model_kwargs),
        inference=InferenceSettings(model_file_path=DEFAULT_MODEL),
        logging=LoggingSettings(level="INFO", log_file="logs/pipeline.log"),
    )
