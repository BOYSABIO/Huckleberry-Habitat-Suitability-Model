"""Data I/O — load and save pipeline datasets (no transforms)."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import DataSettings

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and save GBIF occurrence data and pipeline CSV artifacts."""

    def __init__(
        self,
        data_settings: Optional[DataSettings] = None,
        data_dir: str = "data",
    ):
        if data_settings is not None:
            self.processed_dir = Path(data_settings.processed_data_path)
            self.enriched_dir = Path(data_settings.enriched_data_path)
            self.raw_dir = Path(data_settings.raw_data_path).parent
        else:
            base = Path(data_dir)
            self.raw_dir = base / "raw"
            self.processed_dir = base / "processed"
            self.enriched_dir = base / "enriched"

        for dir_path in (self.raw_dir, self.processed_dir, self.enriched_dir):
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_gbif_occurrences(self, filepath: Optional[str] = None) -> pd.DataFrame:
        if filepath is None:
            filepath = self.raw_dir / "occurrence.txt"
        logger.info("Loading GBIF occurrences from %s", filepath)
        df = pd.read_csv(filepath, sep="\t", low_memory=False)
        logger.info("Loaded %s occurrence records", len(df))
        required = ["decimalLatitude", "decimalLongitude", "eventDate"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    def load_enriched_data(self, filename: str) -> pd.DataFrame:
        filepath = self.enriched_dir / filename
        logger.info("Loading enriched data from %s", filepath)
        return pd.read_csv(filepath)

    def load_training_dataset(self, filepath: str) -> pd.DataFrame:
        """Load a pre-enriched training CSV (any path)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Training dataset not found: {path}")
        logger.info("Loading training dataset from %s", path)
        df = pd.read_csv(path)
        logger.info("Loaded %s training records", len(df))
        return df

    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        filepath = self.processed_dir / filename
        logger.info("Saving processed data to %s", filepath)
        df.to_csv(filepath, index=False)
        logger.info("Saved %s records to %s", len(df), filepath)

    def save_enriched_data(self, df: pd.DataFrame, filename: str) -> None:
        filepath = self.enriched_dir / filename
        logger.info("Saving enriched data to %s", filepath)
        df.to_csv(filepath, index=False)
        logger.info("Saved %s records to %s", len(df), filepath)
