import pandas as pd
import numpy as np
import xarray as xr
import os
from tqdm import tqdm
import pystac_client
import planetary_computer
import time

# ----------- CONFIG -----------
INPUT_CSV = "./data/processed/huckleberry_data_spencer.csv"    ## put your csv file here
OUTPUT_CSV = "./data/enriched/final_df_spencer.csv"

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
            chunks={"time": 365, "lat": 100, "lon": 100}
        )
        print("✅ Dataset loaded.")
        return ds
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        raise

# Load dataset
ds = load_dataset()

# Load input data
df = pd.read_csv(INPUT_CSV)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.reset_index(drop=True)

# Resume from where we left off
if os.path.exists(OUTPUT_CSV):
    existing = pd.read_csv(OUTPUT_CSV)
    start_idx = len(existing)
    print(f"🔁 Resuming from index {start_idx}")
else:
    existing = pd.DataFrame()
    start_idx = 0

# Dataset bounds
lat_min, lat_max = ds.lat.values.min(), ds.lat.values.max()
lon_min, lon_max = ds.lon.values.min(), ds.lon.values.max()
time_min, time_max = pd.to_datetime(ds.time.values.min()), pd.to_datetime(ds.time.values.max())

# Extraction loop
for i in tqdm(range(start_idx, len(df)), desc="Extracting row by row"):
    row = df.iloc[i]
    lat = row["decimalLatitude"]
    lon = row["decimalLongitude"]
    date = row["date"]

    try:
        print(f"\n🔍 Row {i}: lat={lat:.5f}, lon={lon:.5f}, date={date.date()}")

        # Basic validation
        if pd.isna(lat) or pd.isna(lon) or pd.isna(date):
            raise ValueError("Missing lat/lon/date")
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max and time_min <= date <= time_max):
            raise ValueError("Out of gridMET bounds")

        # Get nearest index
        lat_idx = np.abs(ds.lat.values - lat).argmin()
        lon_idx = np.abs(ds.lon.values - lon).argmin()
        time_idx = np.abs(ds.time.values - np.datetime64(date)).argmin()

        # Snapped coordinate metadata
        gridmet_lat = ds.lat.values[lat_idx]
        gridmet_lon = ds.lon.values[lon_idx]
        gridmet_date = pd.to_datetime(ds.time.values[time_idx])

        print(f"📍 gridMET point: lat={gridmet_lat:.5f}, lon={gridmet_lon:.5f}, date={gridmet_date.date()}")

        # Load cube and extract data
        data_cube = ds[VARS_TO_EXTRACT].isel(
            lat=lat_idx, lon=lon_idx, time=time_idx
        ).compute()

        env_data = {
            "gridmet_lat": gridmet_lat,
            "gridmet_lon": gridmet_lon,
            "gridmet_date": gridmet_date
        }
        env_data.update({var: data_cube[var].values.item() for var in VARS_TO_EXTRACT})

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
