"""
Data preprocessing module for huckleberry habitat prediction.

This module handles the initial cleaning and filtering of GBIF occurrence data,
including coordinate validation, date processing, and geographic filtering.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Preprocesses raw GBIF occurrence data for analysis.
    """
    def __init__(self):
        self.us_bounds = {
            'lat_min': 24.0,
            'lat_max': 71.0,
            'lon_min': -180.0,
            'lon_max': -66.0
        }
        self.required_columns = [
            'decimalLatitude', 'decimalLongitude', 'eventDate',
            'countryCode', 'stateProvince', 'county'
        ]

    def clean_occurrence_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning process...")
        original_count = len(df)
        df = self._basic_cleaning(df)
        df = self._filter_geographic_bounds(df)
        df = self._filter_by_country(df)
        df = self._clean_coordinates(df)
        df = self._process_dates(df)
        df = self._remove_duplicates(df)
        df = self._select_columns(df)
        final_count = len(df)
        logger.info(f"Cleaning complete: {original_count} -> {final_count} records "
                    f"({final_count/original_count*100:.1f}% kept)")
        return df

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Performing basic cleaning...")
        df = df.dropna(how='all')
        df.columns = df.columns.str.lower()
        df['countrycode'] = df['countrycode'].fillna('Unknown')
        df['stateprovince'] = df['stateprovince'].fillna('Unknown')
        df['county'] = df['county'].fillna('Unknown')
        return df

    def _filter_geographic_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering by geographic bounds...")
        df['decimallatitude'] = pd.to_numeric(df['decimallatitude'], errors='coerce')
        df['decimallongitude'] = pd.to_numeric(df['decimallongitude'], errors='coerce')
        df = df.dropna(subset=['decimallatitude', 'decimallongitude'])
        mask = (
            (df['decimallatitude'] >= self.us_bounds['lat_min']) &
            (df['decimallatitude'] <= self.us_bounds['lat_max']) &
            (df['decimallongitude'] >= self.us_bounds['lon_min']) &
            (df['decimallongitude'] <= self.us_bounds['lon_max'])
        )
        filtered_df = df[mask].copy()
        logger.info(f"Geographic filtering: {len(df)} -> {len(filtered_df)} records")
        return filtered_df

    def _filter_by_country(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering by country (US only)...")
        us_mask = df['countrycode'].str.upper().isin(['US', 'USA', 'UNITED STATES'])
        filtered_df = df[us_mask].copy()
        logger.info(f"Country filtering: {len(df)} -> {len(filtered_df)} records")
        return filtered_df

    def _clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning coordinates...")
        invalid_lat = (
            (df['decimallatitude'] == 0) |
            (df['decimallatitude'] > 90) |
            (df['decimallatitude'] < -90)
        )
        invalid_lon = (
            (df['decimallongitude'] == 0) |
            (df['decimallongitude'] > 180) |
            (df['decimallongitude'] < -180)
        )
        df = df[~(invalid_lat | invalid_lon)]
        df['decimallatitude'] = df['decimallatitude'].round(5)
        df['decimallongitude'] = df['decimallongitude'].round(5)
        logger.info(f"Coordinate cleaning: {len(df)} records remaining")
        return df

    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Processing dates...")
        df['eventdate'] = pd.to_datetime(df['eventdate'], errors='coerce')
        df = df.dropna(subset=['eventdate'])
        df['year'] = df['eventdate'].dt.year
        df['month'] = df['eventdate'].dt.month
        df['day'] = df['eventdate'].dt.day
        df = df[df['year'] >= 1900]
        logger.info(f"Date processing: {len(df)} records remaining")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing duplicates...")
        original_count = len(df)
        df = df.drop_duplicates()
        df = df.drop_duplicates(subset=[
            'decimallatitude', 'decimallongitude', 'eventdate'
        ])
        logger.info(f"Duplicate removal: {original_count} -> {len(df)} records")
        return df

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Selecting relevant columns...")
        columns_to_keep = [
            'decimallatitude', 'decimallongitude', 'eventdate',
            'countrycode', 'stateprovince', 'county', 'locality',
            'verbatimlocality', 'municipality', 'year', 'month', 'day'
        ]
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[existing_columns]
        logger.info(f"Selected {len(existing_columns)} columns")
        return df

    def create_pseudo_absences(self, df: pd.DataFrame, ratio: float = 3.0) -> pd.DataFrame:
        logger.info(f"Creating pseudo-absences with ratio {ratio}:1")
        lat_min, lat_max = df['decimallatitude'].min(), df['decimallatitude'].max()
        lon_min, lon_max = df['decimallongitude'].min(), df['decimallongitude'].max()
        n_presences = len(df)
        n_absences = int(n_presences * ratio)
        np.random.seed(42)
        pseudo_lats = np.random.uniform(lat_min, lat_max, n_absences)
        pseudo_lons = np.random.uniform(lon_min, lon_max, n_absences)
        pseudo_absences = pd.DataFrame({
            'decimallatitude': pseudo_lats,
            'decimallongitude': pseudo_lons,
            'eventdate': df['eventdate'].sample(n=n_absences, replace=True).values,
            'countrycode': 'US',
            'stateprovince': 'Unknown',
            'county': 'Unknown',
            'locality': 'Pseudo-absence',
            'verbatimlocality': 'Pseudo-absence',
            'municipality': 'Unknown',
            'year': df['year'].sample(n=n_absences, replace=True).values,
            'month': df['month'].sample(n=n_absences, replace=True).values,
            'day': df['day'].sample(n=n_absences, replace=True).values
        })
        df['occurrence'] = 1
        pseudo_absences['occurrence'] = 0
        combined_df = pd.concat([df, pseudo_absences], ignore_index=True)
        logger.info(f"Created {n_absences} pseudo-absences, total dataset: {len(combined_df)} records")
        return combined_df

    def get_cleaning_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
        summary = {
            'original_records': len(original_df),
            'cleaned_records': len(cleaned_df),
            'records_removed': len(original_df) - len(cleaned_df),
            'retention_rate': len(cleaned_df) / len(original_df) * 100,
            'geographic_bounds': self.us_bounds,
            'date_range': {
                'min_date': cleaned_df['eventdate'].min(),
                'max_date': cleaned_df['eventdate'].max()
            }
        }
        return summary 