"""
Environmental data extraction module for huckleberry habitat prediction.

This module handles the extraction and integration of environmental data:
- GridMET climate data (temperature, precipitation, humidity, etc.)
- Elevation data from Open-Elevation API
- Soil pH data from SoilGrids API
"""

import pandas as pd
import numpy as np
import xarray as xr
import requests
import time
import logging
from tqdm import tqdm
import pystac_client
import planetary_computer

logger = logging.getLogger(__name__)

class EnvironmentalDataExtractor:
    """
    Extracts and integrates environmental data for habitat modeling.
    """
    
    def __init__(self):
        self.gridmet_vars = [
            "air_temperature",
            "precipitation_amount", 
            "specific_humidity",
            "relative_humidity",
            "mean_vapor_pressure_deficit",
            "potential_evapotranspiration",
            "surface_downwelling_shortwave_flux_in_air",
            "wind_speed"
        ]
        self.elevation_api_url = "https://api.open-elevation.com/api/v1/lookup"
        self.soilgrid_api_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        self.request_delay = 1  # seconds between API calls
        
    def load_gridmet_dataset(self):
        """Load the GridMET dataset from Planetary Computer."""
        logger.info("Connecting to Planetary Computer...")
        try:
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )
            asset = catalog.get_collection("gridmet").assets["zarr-abfs"]
            ds = xr.open_zarr(
                asset.href,
                storage_options=asset.extra_fields["xarray:storage_options"],
                **asset.extra_fields["xarray:open_kwargs"],
                chunks="auto"
            )
            logger.info("GridMET dataset loaded successfully")
            return ds
        except Exception as e:
            logger.error(f"Failed to load GridMET dataset: {e}")
            raise
    
    def extract_gridmet_data(self, df, target_date=None):
        """
        Extract GridMET climate data for each location and date.
        
        Args:
            df: DataFrame with decimalLatitude, decimalLongitude, and datetime columns
            target_date: Specific date to use for GridMET data (if None, uses datetime column or latest)
            
        Returns:
            DataFrame with GridMET data added (only records within bounds and with valid data)
        """
        logger.info("Extracting GridMET climate data...")
        
        try:
            # Load GridMET dataset
            ds = self.load_gridmet_dataset()
            
            # Get dataset bounds
            lat_min, lat_max = ds.lat.values.min(), ds.lat.values.max()
            lon_min, lon_max = ds.lon.values.min(), ds.lon.values.max()
            time_min, time_max = pd.to_datetime(ds.time.values.min()), pd.to_datetime(ds.time.values.max())
            
            logger.info(f"GridMET bounds: Lat {lat_min:.3f}-{lat_max:.3f}, Lon {lon_min:.3f}-{lon_max:.3f}")
            logger.info(f"Time range: {time_min.date()} to {time_max.date()}")
            
            # Filter to records within GridMET bounds
            df_filtered = df.copy()
            
            # Handle datetime column - prioritize target_date if provided
            if target_date is not None:
                target_date = pd.to_datetime(target_date)
                logger.info(f"Using specified target date: {target_date}")
                
                # Validate target date is within GridMET bounds
                if target_date < time_min or target_date > time_max:
                    logger.warning(f"Target date {target_date} is outside GridMET time range. "
                                 f"Using latest available date instead.")
                    target_date = pd.to_datetime(ds.time.values.max())
                    logger.info(f"Using latest GridMET date: {target_date}")
                
                df_filtered['datetime'] = target_date
            elif 'datetime' not in df_filtered.columns:
                logger.info("No datetime column provided, using latest available GridMET date")
                latest_date = pd.to_datetime(ds.time.values.max())
                df_filtered['datetime'] = latest_date
                logger.info(f"Using latest GridMET date: {latest_date}")
            else:
                df_filtered['datetime'] = pd.to_datetime(df_filtered['datetime'])
            
            # Spatial filtering
            spatial_mask = (
                (df_filtered['decimalLatitude'] >= lat_min) &
                (df_filtered['decimalLatitude'] <= lat_max) &
                (df_filtered['decimalLongitude'] >= lon_min) &
                (df_filtered['decimalLongitude'] <= lon_max)
            )
            
            # Temporal filtering (only if datetime was provided)
            if 'datetime' in df.columns:
                temporal_mask = (
                    (df_filtered['datetime'] >= time_min) &
                    (df_filtered['datetime'] <= time_max)
                )
                valid_mask = spatial_mask & temporal_mask
            else:
                # If using latest date, only check spatial bounds
                valid_mask = spatial_mask
            
            df_valid = df_filtered[valid_mask].copy()
            
            logger.info(f"Records within GridMET bounds: {len(df_valid)}/{len(df)}")
            
            if len(df_valid) == 0:
                logger.warning("No records within GridMET bounds")
                return pd.DataFrame()  # Return empty DataFrame instead of original
            
            # Extract data for each record
            gridmet_data = []
            
            for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Extracting GridMET data"):
                try:
                    lat = row['decimalLatitude']
                    lon = row['decimalLongitude']
                    date = row['datetime']
                    
                    # Find nearest grid points
                    lat_idx = np.abs(ds.lat.values - lat).argmin()
                    lon_idx = np.abs(ds.lon.values - lon).argmin()
                    time_idx = np.abs(ds.time.values - np.datetime64(date)).argmin()
                    
                    # Extract data
                    data_cube = ds[self.gridmet_vars].isel(
                        lat=lat_idx, lon=lon_idx, time=time_idx
                    ).compute()
                    
                    # Check for NaN in any variable
                    values = {var: data_cube[var].values.item() for var in self.gridmet_vars}
                    if any(pd.isna(val) for val in values.values()):
                        logger.warning(f"Skipping row {idx}: GridMET data contains NaN values.")
                        continue
                    
                    # Create record with GridMET data
                    record = row.to_dict()
                    record.update({
                        'gridmet_lat': ds.lat.values[lat_idx],
                        'gridmet_lon': ds.lon.values[lon_idx],
                        'gridmet_date': pd.to_datetime(ds.time.values[time_idx])
                    })
                    record.update(values)
                    
                    gridmet_data.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract GridMET data for row {idx}: {e}")
                    # Skip this record entirely instead of adding it back without data
                    continue
            
            # Create enriched DataFrame
            enriched_df = pd.DataFrame(gridmet_data)
            
            logger.info(f"GridMET extraction complete: {len(enriched_df)} records (with valid data)")
            return enriched_df
            
        except Exception as e:
            logger.error(f"Failed to extract GridMET data: {e}")
            raise
    
    def get_elevation(self, lat, lon):
        """Query Open-Elevation API for elevation data."""
        try:
            response = requests.get(self.elevation_api_url, params={"locations": f"{lat},{lon}"})
            if response.status_code == 200:
                elevation = response.json()["results"][0]["elevation"]
                return elevation
            else:
                logger.warning(f"Elevation API error: {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Failed to get elevation for {lat}, {lon}: {e}")
            return None
    
    def add_elevation_data(self, df):
        """
        Add elevation data from Open-Elevation API.
        
        Args:
            df: DataFrame with decimalLatitude and decimalLongitude columns
            
        Returns:
            DataFrame with elevation column added
        """
        logger.info("Adding elevation data...")
        
        # Add elevation column if it doesn't exist
        if 'elevation' not in df.columns:
            df['elevation'] = np.nan
        
        # Get elevation for records that don't have it
        missing_elevation = df['elevation'].isna()
        records_to_process = df[missing_elevation].copy()
        
        if len(records_to_process) == 0:
            logger.info("All records already have elevation data")
            return df
        
        logger.info(f"Processing elevation for {len(records_to_process)} records...")
        
        for idx, row in tqdm(records_to_process.iterrows(), total=len(records_to_process), desc="Adding elevation"):
            lat = row['decimalLatitude']
            lon = row['decimalLongitude']
            elevation = self.get_elevation(lat, lon)
            df.loc[idx, 'elevation'] = elevation
            
            # Be polite to the API
            time.sleep(self.request_delay)
        
        logger.info(f"Elevation data added: {df['elevation'].notna().sum()}/{len(df)} records")
        return df
    
    def get_soil_ph(self, lat, lon, retries=3, base_delay=2):
        """Fetch soil pH from SoilGrids API with retry logic."""
        url = (
            f"{self.soilgrid_api_url}?lat={lat}&lon={lon}"
            f"&property=phh2o&depth=15-30cm&value=mean"
        )
        
        for attempt in range(retries):
            try:
                response = requests.get(url)
                if response.status_code == 429:
                    wait = base_delay * (2 ** attempt)
                    logger.info(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                response.raise_for_status()
                data = response.json()
                layers = data.get("properties", {}).get("layers", [])
                
                for layer in layers:
                    if layer["name"] == "phh2o":
                        values = layer["depths"][0]["values"]
                        raw_ph = values.get("mean")
                        if raw_ph is not None:
                            return round(raw_ph / 10.0, 2)  # Convert to pH scale
                return None
                
            except Exception as e:
                logger.warning(f"Soil pH error at ({lat}, {lon}): {e}")
                if attempt < retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        
        return None
    
    def add_soil_data(self, df):
        """
        Add soil pH data from SoilGrids API.
        
        Args:
            df: DataFrame with decimalLatitude and decimalLongitude columns
            
        Returns:
            DataFrame with soil_ph column added
        """
        logger.info("Adding soil pH data...")
        
        # Add soil_ph column if it doesn't exist
        if 'soil_ph' not in df.columns:
            df['soil_ph'] = np.nan
        
        # Get soil pH for records that don't have it
        missing_soil = df['soil_ph'].isna()
        records_to_process = df[missing_soil].copy()
        
        if len(records_to_process) == 0:
            logger.info("All records already have soil pH data")
            return df
        
        logger.info(f"Processing soil pH for {len(records_to_process)} records...")
        
        for idx, row in tqdm(records_to_process.iterrows(), total=len(records_to_process), desc="Adding soil pH"):
            lat = row['decimalLatitude']
            lon = row['decimalLongitude']
            soil_ph = self.get_soil_ph(lat, lon)
            df.loc[idx, 'soil_ph'] = soil_ph
            
            # Be polite to the API
            time.sleep(self.request_delay)
        
        logger.info(f"Soil pH data added: {df['soil_ph'].notna().sum()}/{len(df)} records")
        return df
    
    def create_environmental_features(self, df):
        """
        Create derived environmental features from the raw data.
        
        Args:
            df: DataFrame with environmental variables
            
        Returns:
            DataFrame with additional derived features
        """
        logger.info("Creating derived environmental features...")
        
        # Temperature features
        if 'air_temperature' in df.columns:
            df['temp_celsius'] = df['air_temperature'] - 273.15  # Convert K to °C
            df['temp_fahrenheit'] = df['temp_celsius'] * 9/5 + 32  # Convert to °F
        
        # Precipitation features
        if 'precipitation_amount' in df.columns:
            df['precip_mm'] = df['precipitation_amount'] * 1000  # Convert m to mm
            df['precip_inches'] = df['precip_mm'] / 25.4  # Convert mm to inches
        
        # Humidity features
        if 'relative_humidity' in df.columns:
            df['rh_percent'] = df['relative_humidity'] * 100  # Convert to percentage
        
        # Wind features
        if 'wind_speed' in df.columns:
            df['wind_mph'] = df['wind_speed'] * 2.237  # Convert m/s to mph
        
        # Elevation features
        if 'elevation' in df.columns:
            df['elevation_m'] = df['elevation']  # Keep in meters
            df['elevation_ft'] = df['elevation'] * 3.28084  # Convert to feet
        
        logger.info(f"Created {len([col for col in df.columns if col not in ['decimalLatitude', 'decimalLongitude', 'datetime', 'occurrence']])} environmental features")
        return df 