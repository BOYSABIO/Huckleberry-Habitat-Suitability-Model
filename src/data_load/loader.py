"""
Data loader module for huckleberry habitat prediction.

This module handles loading raw data from various sources including:
- GBIF occurrence data
- Environmental datasets
- Climate data
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads and validates data from various sources.
    """
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.enriched_dir = self.data_dir / "enriched"
        # Ensure directories exist
        for dir_path in [self.raw_dir, self.processed_dir, self.enriched_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_gbif_occurrences(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load GBIF occurrence data.
        Args:
            filepath: Path to the occurrence file. If None, uses default location.
        Returns:
            DataFrame with occurrence data
        """
        if filepath is None:
            filepath = self.raw_dir / "occurrence.txt"
        logger.info(f"Loading GBIF occurrences from {filepath}")
        try:
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
            logger.info(f"Loaded {len(df)} occurrence records")
            required_columns = ['decimalLatitude', 'decimalLongitude', 'eventDate']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            return df
        except Exception as e:
            logger.error(f"Failed to load GBIF data: {e}")
            raise

    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from the processed directory.
        Args:
            filename: Name of the file to load
        Returns:
            DataFrame with processed data
        """
        filepath = self.processed_dir / filename
        logger.info(f"Loading processed data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} processed records")
            return df
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise

    def load_enriched_data(self, filename: str) -> pd.DataFrame:
        """
        Load enriched data from the enriched directory.
        Args:
            filename: Name of the file to load
        Returns:
            DataFrame with enriched data
        """
        filepath = self.enriched_dir / filename
        logger.info(f"Loading enriched data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} enriched records")
            return df
        except Exception as e:
            logger.error(f"Failed to load enriched data: {e}")
            raise

    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to the processed directory.
        Args:
            df: DataFrame to save
            filename: Name of the file to save
        """
        filepath = self.processed_dir / filename
        logger.info(f"Saving processed data to {filepath}")
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} records to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise

    def save_enriched_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save enriched data to the enriched directory.
        Args:
            df: DataFrame to save
            filename: Name of the file to save
        """
        filepath = self.enriched_dir / filename
        logger.info(f"Saving enriched data to {filepath}")
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} records to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save enriched data: {e}")
            raise

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the dataset.
        Args:
            df: DataFrame to summarize
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        return summary

    def validate_data(self, df: pd.DataFrame, expected_columns: list = None) -> bool:
        """
        Validate that the data meets basic requirements.
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating data...")
        if len(df) == 0:
            logger.error("DataFrame is empty")
            return False
        if expected_columns:
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            logger.warning(f"Columns with all null values: {null_columns}")
        logger.info("Data validation passed")
        return True 