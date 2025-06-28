"""
Data validation module for huckleberry pipeline.
"""

import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame, expected_columns: List[str] = None) -> bool:
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