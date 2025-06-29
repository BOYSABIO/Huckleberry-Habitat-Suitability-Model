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
    
    # Check if DataFrame is empty
    if len(df) == 0:
        logger.error("DataFrame is empty")
        return False
    
    # Check for required columns
    if expected_columns:
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
    
    # Check for temporal columns (year, month, day)
    temporal_columns = ['year', 'month', 'day']
    missing_temporal = [col for col in temporal_columns if col not in df.columns]
    if missing_temporal:
        logger.error(f"Missing temporal columns: {missing_temporal}")
        return False
    
    # Validate temporal data
    for col in temporal_columns:
        if col in df.columns:
            # Check for null values
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.error(f"Column '{col}' has {null_count} null values")
                return False
            
            # Check for valid ranges
            if col == 'year':
                invalid_years = df[col][(df[col] < 1800) | (df[col] > 2025)]
                if len(invalid_years) > 0:
                    logger.error(f"Column '{col}' has {len(invalid_years)} invalid years (outside 1800-2025)")
                    return False
            elif col == 'month':
                invalid_months = df[col][(df[col] < 1) | (df[col] > 12)]
                if len(invalid_months) > 0:
                    logger.error(f"Column '{col}' has {len(invalid_months)} invalid months (outside 1-12)")
                    return False
            elif col == 'day':
                invalid_days = df[col][(df[col] < 1) | (df[col] > 31)]
                if len(invalid_days) > 0:
                    logger.error(f"Column '{col}' has {len(invalid_days)} invalid days (outside 1-31)")
                    return False
    
    # Check for coordinate columns if they exist
    coord_columns = ['decimalLatitude', 'decimalLongitude']
    for col in coord_columns:
        if col in df.columns:
            # Check for valid coordinate ranges
            if col == 'decimalLatitude':
                invalid_lat = df[col][(df[col] < -90) | (df[col] > 90)]
                if len(invalid_lat) > 0:
                    logger.error(f"Column '{col}' has {len(invalid_lat)} invalid latitudes (outside -90 to 90)")
                    return False
            elif col == 'decimalLongitude':
                invalid_lon = df[col][(df[col] < -180) | (df[col] > 180)]
                if len(invalid_lon) > 0:
                    logger.error(f"Column '{col}' has {len(invalid_lon)} invalid longitudes (outside -180 to 180)")
                    return False
    
    # Check for columns with all null values (warning only)
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        logger.warning(f"Columns with all null values: {null_columns}")
    
    logger.info("Data validation passed")
    return True 