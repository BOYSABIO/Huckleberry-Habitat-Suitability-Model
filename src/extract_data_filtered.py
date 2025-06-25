import pandas as pd
import numpy as np
import xarray as xr
import os
from tqdm import tqdm
import pystac_client
import planetary_computer
import time

# ----------- CONFIG -----------
INPUT_CSV = "./data/processed/date_location_occurrances.csv"
OUTPUT_CSV = "./data/enriched/filtered_df_spencer.csv"

VARS_TO_EXTRACT = [
    "air_temperature",
    "precipitation_amount",
    "specific_humidity",
    "relative_humidity",
    "mean_vapor_pressure_deficit",
    "potential_evapotranspiration",
    "surface_downwelling_shortwave_flux_in_air",
    "wind_speed"
]

RETRY_DELAY_SECONDS = 10
# ------------------------------


def load_dataset():
    """Load the gridMET dataset and return it."""
    try:
        print("🔄 Connecting to Planetary Computer...")
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
        print("✅ Dataset loaded.")
        return ds
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        raise


def filter_data(df, lat_min, lat_max, lon_min, lon_max, time_min, time_max):
    """Filter data to only include points within gridMET bounds."""
    print("🔍 Filtering data to gridMET bounds...")
    
    # Convert datetime column
    df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
    
    # Create bounds masks
    lat_in_bounds = (df['decimalLatitude'] >= lat_min) & (df['decimalLatitude'] <= lat_max)
    lon_in_bounds = (df['decimalLongitude'] >= lon_min) & (df['decimalLongitude'] <= lon_max)
    time_in_bounds = (df['date'] >= time_min) & (df['date'] <= time_max)
    
    # Combine all bounds
    all_in_bounds = lat_in_bounds & lon_in_bounds & time_in_bounds
    
    # Filter the dataframe
    filtered_df = df[all_in_bounds].copy().reset_index(drop=True)
    
    print(f"📊 Filtering Results:")
    print(f"   Original data: {len(df)} rows")
    print(f"   Filtered data: {len(filtered_df)} rows")
    print(f"   Removed: {len(df) - len(filtered_df)} rows")
    print(f"   Kept: {len(filtered_df)/len(df)*100:.1f}%")
    
    return filtered_df


def main():
    # Load dataset
    ds = load_dataset()
    
    # Get bounds
    lat_min, lat_max = ds.lat.values.min(), ds.lat.values.max()
    lon_min, lon_max = ds.lon.values.min(), ds.lon.values.max()
    time_min, time_max = pd.to_datetime(ds.time.values.min()), pd.to_datetime(
        ds.time.values.max()
    )
    
    print("📊 gridMET Dataset Bounds:")
    print(f"   Latitude: {lat_min:.3f} to {lat_max:.3f}")
    print(f"   Longitude: {lon_min:.3f} to {lon_max:.3f}")
    print(f"   Time: {time_min.date()} to {time_max.date()}")
    
    # Load and filter input data
    print("\n📂 Loading and filtering data...")
    df = pd.read_csv(INPUT_CSV)
    filtered_df = filter_data(df, lat_min, lat_max, lon_min, lon_max, time_min, time_max)
    
    if len(filtered_df) == 0:
        print("❌ No data points within bounds. Exiting.")
        return
    
    # Resume from where we left off
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        start_idx = len(existing)
        print(f"🔁 Resuming from index {start_idx}")
    else:
        existing = pd.DataFrame()
        start_idx = 0
    
    # Extraction loop
    for i in tqdm(range(start_idx, len(filtered_df)), desc="Extracting row by row"):
        row = filtered_df.iloc[i]
        lat = row["decimalLatitude"]
        lon = row["decimalLongitude"]
        date = row["date"]
        
        try:
            print(f"\n🔍 Row {i}: lat={lat:.5f}, lon={lon:.5f}, date={date.date()}")
            
            # Get nearest index
            lat_idx = np.abs(ds.lat.values - lat).argmin()
            lon_idx = np.abs(ds.lon.values - lon).argmin()
            time_idx = np.abs(ds.time.values - np.datetime64(date)).argmin()
            
            # Snapped coordinate metadata
            gridmet_lat = ds.lat.values[lat_idx]
            gridmet_lon = ds.lon.values[lon_idx]
            gridmet_date = pd.to_datetime(ds.time.values[time_idx])
            
            print(f"📍 gridMET point: lat={gridmet_lat:.5f}, "
                  f"lon={gridmet_lon:.5f}, date={gridmet_date.date()}")
            
            # Load cube and extract data
            data_cube = ds[VARS_TO_EXTRACT].isel(
                lat=lat_idx, lon=lon_idx, time=time_idx
            ).compute()
            
            env_data = {
                "gridmet_lat": gridmet_lat,
                "gridmet_lon": gridmet_lon,
                "gridmet_date": gridmet_date
            }
            env_data.update({var: data_cube[var].values.item() 
                            for var in VARS_TO_EXTRACT})
            
            # Merge and save
            row_out = pd.concat([row, pd.Series(env_data)]).to_frame().T
            if not os.path.exists(OUTPUT_CSV):
                row_out.to_csv(OUTPUT_CSV, index=False)
            else:
                row_out.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
            
            print("✅ Row saved.")
            
        except Exception as e:
            print(f"❌ Failed at row {i}: {e}")
            print("🔄 Reconnecting and retrying after delay...")
            try:
                time.sleep(RETRY_DELAY_SECONDS)
                ds = load_dataset()
                print("✅ Reconnected. Retrying row...")
                continue  # retry same row
            except Exception as e2:
                print(f"🛑 Reconnection failed. Stopping. Error: {e2}")
                break


if __name__ == "__main__":
    main() 