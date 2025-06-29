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
        # GridMET bounds based on actual dataset coverage
        self.gridmet_bounds = {
            'lat_min': 25.07,
            'lat_max': 49.4,
            'lon_min': -124.8,
            'lon_max': -67.06,
            'time_min': pd.Timestamp('1979-01-01'),
            'time_max': pd.Timestamp('2020-12-31')
        }
        self.required_columns = [
            'decimalLatitude', 'decimalLongitude', 'eventDate',
            'countryCode', 'stateProvince', 'county'
        ]

    def clean_occurrence_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the occurrence data, but do NOT select only essential columns yet.
        Essential columns should be selected at the very end, after geocoding and manual geocoding.
        GridMET filtering should be done after geocoding when coordinates are available.
        """
        logger.info("Starting data cleaning process...")
        original_count = len(df)
        df = self._basic_cleaning(df)
        df = self._filter_geographic_bounds(df)
        df = self._filter_by_country(df)
        df = self._clean_coordinates(df)
        df = self._process_dates(df)
        # Ensure 'datetime' column is present and valid after date processing
        if 'datetime' not in df.columns or df['datetime'].isna().all():
            logger.warning("No valid 'datetime' column found after date processing. Attempting to create from year/month/day.")
            if all(col in df.columns for col in ['year', 'month', 'day']):
                try:
                    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
                    logger.info(f"Created 'datetime' column from year/month/day: {df['datetime'].notna().sum()} valid records")
                except Exception as e:
                    logger.error(f"Failed to create 'datetime' column from year/month/day: {e}")
            else:
                logger.error("Cannot create 'datetime' column: missing year/month/day columns.")
        # GridMET filtering moved to after geocoding
        df = self._remove_duplicates(df)
        # Do NOT select columns here; do it after geocoding/manual geocoding
        final_count = len(df)
        logger.info(f"Cleaning complete: {original_count} -> {final_count} records "
                    f"({final_count/original_count*100:.1f}% kept)")
        df['occurrence'] = 1
        return df

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Performing basic cleaning...")
        df = df.dropna(how='all')
        df['countryCode'] = df['countryCode'].fillna('Unknown')
        df['stateProvince'] = df['stateProvince'].fillna('Unknown')
        df['county'] = df['county'].fillna('Unknown')
        # Clean state names to remove non-ASCII characters
        df = self._clean_state_names(df)
        return df

    def _clean_state_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean state names by removing non-ASCII characters."""
        logger.info("Cleaning state names...")
        original_states = df['stateProvince'].unique()
        
        # Remove non-ASCII characters from state names
        df['stateProvince'] = df['stateProvince'].astype(str).apply(
            lambda x: ''.join(char for char in x if ord(char) < 128)
        )
        
        # Clean up common issues
        df['stateProvince'] = df['stateProvince'].str.replace('  ', ' ').str.strip()
        
        cleaned_states = df['stateProvince'].unique()
        logger.info(f"State name cleaning: {len(original_states)} -> {len(cleaned_states)} unique states")
        
        # Log any significant changes
        for original in original_states:
            if pd.notna(original):
                cleaned = df[df['stateProvince'].str.contains(original.split()[0], na=False)]['stateProvince'].iloc[0] if len(df[df['stateProvince'].str.contains(original.split()[0], na=False)]) > 0 else None
                if cleaned and cleaned != original:
                    logger.info(f"  Cleaned state: '{original}' -> '{cleaned}'")
        
        return df

    def _filter_geographic_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering by geographic bounds...")
        df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
        df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')
        
        # Only filter records that have coordinates and are outside US bounds
        # Keep records without coordinates for geocoding later
        has_coords = df['decimalLatitude'].notna() & df['decimalLongitude'].notna()
        
        if has_coords.any():
            # Filter only the records that have coordinates
            coords_mask = (
                (df['decimalLatitude'] >= self.us_bounds['lat_min']) &
                (df['decimalLatitude'] <= self.us_bounds['lat_max']) &
                (df['decimalLongitude'] >= self.us_bounds['lon_min']) &
                (df['decimalLongitude'] <= self.us_bounds['lon_max'])
            )
            
            # Keep records with valid coordinates OR records without coordinates
            valid_mask = coords_mask | ~has_coords
            filtered_df = df[valid_mask].copy()
            
            records_with_coords = has_coords.sum()
            records_outside_bounds = (has_coords & ~coords_mask).sum()
            records_without_coords = (~has_coords).sum()
            
            logger.info(f"Geographic filtering: {len(df)} -> {len(filtered_df)} records")
            logger.info(f"  - Records with coordinates: {records_with_coords}")
            logger.info(f"  - Records outside US bounds (dropped): {records_outside_bounds}")
            logger.info(f"  - Records without coordinates (kept for geocoding): {records_without_coords}")
        else:
            # No coordinates to filter, keep all records
            filtered_df = df.copy()
            logger.info(f"Geographic filtering: {len(df)} -> {len(filtered_df)} records (no coordinates to filter)")
        
        return filtered_df

    def _filter_by_country(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering by country (US only)...")
        us_mask = df['countryCode'].str.upper().isin(['US', 'USA', 'UNITED STATES'])
        filtered_df = df[us_mask].copy()
        logger.info(f"Country filtering: {len(df)} -> {len(filtered_df)} records")
        return filtered_df

    def _clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning coordinates...")
        
        # Only clean records that have coordinates
        has_coords = df['decimalLatitude'].notna() & df['decimalLongitude'].notna()
        
        if has_coords.any():
            # Check for invalid coordinates only in records that have them
            invalid_lat = (
                (df['decimalLatitude'] == 0) |
                (df['decimalLatitude'] > 90) |
                (df['decimalLatitude'] < -90)
            ) & has_coords
            
            invalid_lon = (
                (df['decimalLongitude'] == 0) |
                (df['decimalLongitude'] > 180) |
                (df['decimalLongitude'] < -180)
            ) & has_coords
            
            # Remove only records with invalid coordinates
            df = df[~(invalid_lat | invalid_lon)]
            
            # Round coordinates for records that have them
            df.loc[has_coords, 'decimalLatitude'] = df.loc[has_coords, 'decimalLatitude'].round(5)
            df.loc[has_coords, 'decimalLongitude'] = df.loc[has_coords, 'decimalLongitude'].round(5)
            
            invalid_count = (invalid_lat | invalid_lon).sum()
            logger.info(f"Coordinate cleaning: {invalid_count} invalid coordinates removed, {len(df)} records remaining")
        else:
            logger.info(f"Coordinate cleaning: no coordinates to clean, {len(df)} records remaining")
        
        return df

    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Processing dates...")
        
        # Strategy 1: Try eventDate first (most reliable)
        df['datetime'] = pd.to_datetime(df['eventDate'], errors='coerce')
        valid_from_event = df['datetime'].notna().sum()
        logger.info(f"Extracted dates from eventDate: {valid_from_event} records")
        
        # Strategy 2: For rows where eventDate failed, try year/month/day columns
        mask = df['datetime'].isna()
        if mask.any() and all(col in df.columns for col in ['year', 'month', 'day']):
            df.loc[mask, 'datetime'] = pd.to_datetime({
                'year': df.loc[mask, 'year'],
                'month': df.loc[mask, 'month'],
                'day': df.loc[mask, 'day']
            }, errors='coerce')
            valid_from_ymd = df.loc[mask, 'datetime'].notna().sum()
            logger.info(f"Extracted dates from year/month/day columns: {valid_from_ymd} records")
        
        # Strategy 3: For rows still missing datetime, try startDayOfYear with year
        mask = df['datetime'].isna()
        if mask.any() and 'startDayOfYear' in df.columns:
            def day_of_year_to_date(row):
                try:
                    if pd.notna(row['year']) and pd.notna(row['startDayOfYear']):
                        return pd.Timestamp(year=int(row['year']), month=1, day=1) + pd.Timedelta(days=int(row['startDayOfYear'])-1)
                except:
                    return pd.NaT
                return pd.NaT
            
            df.loc[mask, 'datetime'] = df[mask].apply(day_of_year_to_date, axis=1)
            valid_from_doy = df.loc[mask, 'datetime'].notna().sum()
            logger.info(f"Extracted dates from startDayOfYear: {valid_from_doy} records")
        
        # Strategy 4: For remaining NaN, try verbatimEventDate
        mask = df['datetime'].isna()
        if mask.any() and 'verbatimEventDate' in df.columns:
            df.loc[mask, 'datetime'] = pd.to_datetime(df.loc[mask, 'verbatimEventDate'], errors='coerce')
            valid_from_verbatim = df.loc[mask, 'datetime'].notna().sum()
            logger.info(f"Extracted dates from verbatimEventDate: {valid_from_verbatim} records")
        
        # Strategy 5: As last resort, use dateIdentified
        mask = df['datetime'].isna()
        if mask.any() and 'dateIdentified' in df.columns:
            df.loc[mask, 'datetime'] = pd.to_datetime(df.loc[mask, 'dateIdentified'], errors='coerce')
            valid_from_identified = df.loc[mask, 'datetime'].notna().sum()
            logger.info(f"Extracted dates from dateIdentified: {valid_from_identified} records")
        
        # Drop records that still don't have valid dates
        missing_datetime = df['datetime'].isna()
        if missing_datetime.any():
            records_dropped = missing_datetime.sum()
            df = df[~missing_datetime].copy()
            logger.info(f"Dropped {records_dropped} records without valid dates")
        
        # Extract year, month, day from the clean datetime column
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        
        logger.info(f"Date processing complete: {len(df)} records with clean year/month/day")
        logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing duplicates...")
        original_count = len(df)
        # Only drop duplicates based on gbifID for traceability
        df = df.drop_duplicates(subset=['gbifID'])
        logger.info(f"Duplicate removal: {original_count} -> {len(df)} records (using gbifID)")
        return df

    def create_pseudo_absences(self, df: pd.DataFrame, ratio: float = 3.0) -> pd.DataFrame:
        logger.info(f"Creating pseudo-absences with ratio {ratio}:1")
        lat_min, lat_max = df['decimalLatitude'].min(), df['decimalLatitude'].max()
        lon_min, lon_max = df['decimalLongitude'].min(), df['decimalLongitude'].max()
        n_presences = len(df)
        n_absences = int(n_presences * ratio)
        np.random.seed(42)
        pseudo_lats = np.random.uniform(lat_min, lat_max, n_absences)
        pseudo_lons = np.random.uniform(lon_min, lon_max, n_absences)
        pseudo_absences = pd.DataFrame({
            'decimalLatitude': pseudo_lats,
            'decimalLongitude': pseudo_lons,
            'eventDate': df['eventDate'].sample(n=n_absences, replace=True).values,
            'countryCode': 'US',
            'stateProvince': 'Unknown',
            'county': 'Unknown',
            'locality': 'Pseudo-absence',
            'verbatimLocality': 'Pseudo-absence',
            'municipality': 'Unknown',
            'year': df['year'].sample(n=n_absences, replace=True).values,
            'month': df['month'].sample(n=n_absences, replace=True).values,
            'day': df['day'].sample(n=n_absences, replace=True).values
        })
        
        # Create datetime column from year, month, day
        pseudo_absences['datetime'] = pd.to_datetime(pseudo_absences[['year', 'month', 'day']])
        
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
                'min_year': cleaned_df['year'].min(),
                'max_year': cleaned_df['year'].max(),
                'date_span_years': cleaned_df['year'].max() - cleaned_df['year'].min()
            }
        }
        return summary

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        essential = [
            'gbifID', 'decimalLatitude', 'decimalLongitude',
            'year', 'month', 'day', 'datetime', 'occurrence'
        ]
        available = [col for col in essential if col in df.columns]
        logger.info(f"Selecting essential columns for next pipeline step...")
        logger.info(f"Selected {len(available)} essential columns: {available}")
        return df[available]

    def filter_gridmet_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to gridMET spatial and temporal bounds.
        This should be called AFTER geocoding when coordinates are available.
        """
        logger.info("Filtering to gridMET bounds (post-geocoding)...")
        
        # Convert coordinates to numeric
        df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
        df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')
        
        # Create datetime column for temporal filtering
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
        
        # Only filter records that have coordinates and dates
        has_coords = df['decimalLatitude'].notna() & df['decimalLongitude'].notna()
        has_dates = df['datetime'].notna()
        
        if has_coords.any() and has_dates.any():
            # Spatial filtering
            spatial_mask = (
                (df['decimalLatitude'] >= self.gridmet_bounds['lat_min']) &
                (df['decimalLatitude'] <= self.gridmet_bounds['lat_max']) &
                (df['decimalLongitude'] >= self.gridmet_bounds['lon_min']) &
                (df['decimalLongitude'] <= self.gridmet_bounds['lon_max'])
            )
            
            # Temporal filtering
            temporal_mask = (
                (df['datetime'] >= self.gridmet_bounds['time_min']) &
                (df['datetime'] <= self.gridmet_bounds['time_max'])
            )
            
            # Combine spatial and temporal filtering
            gridmet_mask = spatial_mask & temporal_mask
            
            # Keep records that pass gridMET bounds OR records without coordinates (for geocoding)
            valid_mask = gridmet_mask | ~has_coords
            filtered_df = df[valid_mask].copy()
            
            # Logging statistics
            records_with_coords_dates = (has_coords & has_dates).sum()
            records_outside_spatial = (has_coords & has_dates & ~spatial_mask).sum()
            records_outside_temporal = (has_coords & has_dates & spatial_mask & ~temporal_mask).sum()
            records_kept = gridmet_mask.sum()
            
            logger.info(f"GridMET filtering: {len(df)} -> {len(filtered_df)} records")
            logger.info(f"  - Records with coordinates and dates: {records_with_coords_dates}")
            logger.info(f"  - Records outside spatial bounds (dropped): {records_outside_spatial}")
            logger.info(f"  - Records outside temporal bounds (dropped): {records_outside_temporal}")
            logger.info(f"  - Records within gridMET bounds: {records_kept}")
            logger.info(f"  - Records without coordinates (kept for geocoding): {(~has_coords).sum()}")
        else:
            # No coordinates or dates to filter, keep all records
            filtered_df = df.copy()
            logger.info(f"GridMET filtering: {len(df)} -> {len(filtered_df)} records (no coords/dates to filter)")
        
        return filtered_df

    def filter_gridmet_time_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to gridMET temporal bounds (1979-01-01 to 2020-12-31).
        This should be called after date processing and before geocoding.
        """
        logger.info("Filtering to gridMET temporal bounds...")
        if 'datetime' not in df.columns:
            logger.warning("No 'datetime' column found; skipping temporal filtering.")
            return df
        mask = (
            (df['datetime'] >= self.gridmet_bounds['time_min']) &
            (df['datetime'] <= self.gridmet_bounds['time_max'])
        )
        filtered_df = df[mask].copy()
        logger.info(f"Temporal filtering: {len(df)} -> {len(filtered_df)} records")
        return filtered_df

    # NOTE: Call select_columns(df) at the very end of your pipeline, after geocoding/manual geocoding. 