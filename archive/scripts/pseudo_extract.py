import pandas as pd
import numpy as np
import xarray as xr
import pystac_client
import planetary_computer
from tqdm import tqdm
import os
import time

# ---------------- CONFIG ----------------
INPUT_CSV = "./data/enriched/huckleberries_pseudoabsence.csv"
OUTPUT_CSV = "./data/enriched/HB_PSEUDO.csv"
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
RETRY_DELAY_SECONDS = 5
# ---------------------------------------

def load_dataset():
    print("Connecting to Planetary Computer...")
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
    print("GridMET dataset loaded.")
    return ds

def enrich_and_write_row(row, ds):
    lat = row["decimalLatitude"]
    lon = row["decimalLongitude"]
    date = pd.to_datetime(row["datetime"])
    
    try:
        lat_idx = np.abs(ds.lat.values - lat).argmin()
        lon_idx = np.abs(ds.lon.values - lon).argmin()
        time_idx = np.abs(ds.time.values - np.datetime64(date)).argmin()

        gridmet_lat = ds.lat.values[lat_idx]
        gridmet_lon = ds.lon.values[lon_idx]
        gridmet_date = pd.to_datetime(ds.time.values[time_idx])

        data_cube = ds[VARS_TO_EXTRACT].isel(
            lat=lat_idx, lon=lon_idx, time=time_idx
        ).compute()

        env_data = {
            "gridmet_lat": gridmet_lat,
            "gridmet_lon": gridmet_lon,
            "gridmet_date": gridmet_date
        }
        env_data.update({var: data_cube[var].values.item() for var in VARS_TO_EXTRACT})

        return pd.DataFrame([{**row.to_dict(), **env_data}])
    
    except Exception as e:
        print(f"Failed to fetch GridMET for row (lat={lat}, lon={lon}, date={date.date()}): {e}")
        return None

def main():
    ds = load_dataset()
    df = pd.read_csv(INPUT_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Prepare output
    if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    header_written = os.path.exists(OUTPUT_CSV)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        row = row.copy()
        if row["occurrence"] == 1:
            # Already has GridMET, just write it
            row.to_frame().T.to_csv(OUTPUT_CSV, mode="a", index=False, header=not header_written)
            header_written = True
            continue

        for attempt in range(2):
            enriched_row = enrich_and_write_row(row, ds)
            if enriched_row is not None:
                enriched_row.to_csv(OUTPUT_CSV, mode="a", index=False, header=not header_written)
                header_written = True
                break
            else:
                print("Retrying after reconnect...")
                try:
                    time.sleep(RETRY_DELAY_SECONDS)
                    ds = load_dataset()
                except Exception as e:
                    print(f"Reconnection failed: {e}")
                    break

if __name__ == "__main__":
    main()