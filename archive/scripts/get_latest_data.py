import numpy as np
import pandas as pd
import xarray as xr
import pystac_client
import planetary_computer

# 1. Define Idaho bounding box and grid
lat_min, lat_max = 41.99, 49.00
lon_min, lon_max = -117.24, -111.04
lat_res, lon_res = 0.05, 0.05  # ~5km grid

lat_grid = np.arange(lat_min, lat_max, lat_res)
lon_grid = np.arange(lon_min, lon_max, lon_res)
grid_lat, grid_lon = np.meshgrid(lat_grid, lon_grid, indexing='ij')
grid_points = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])

# 2. Connect to Planetary Computer and load GridMET
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

# 3. Get the most recent date available
latest_time = ds.time.values.max()
print(f"Using latest available date: {pd.to_datetime(latest_time).date()}")

# 4. Extract variables for all grid points
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

# Snap grid points to nearest gridMET grid
def snap_to_grid(val, grid):
    return grid[np.abs(grid - val).argmin()]

snapped_lats = np.array([snap_to_grid(lat, ds.lat.values) for lat in grid_points[:, 0]])
snapped_lons = np.array([snap_to_grid(lon, ds.lon.values) for lon in grid_points[:, 1]])

# Build DataFrame
prediction_df = pd.DataFrame({
    "decimalLatitude": snapped_lats,
    "decimalLongitude": snapped_lons
})

# Extract all variables at once for the latest date
for var in VARS_TO_EXTRACT:
    print(f"Extracting {var}...")
    # Use vectorized selection
    values = ds[var].sel(
        time=latest_time,
        lat=xr.DataArray(snapped_lats, dims="points"),
        lon=xr.DataArray(snapped_lons, dims="points")
    ).values
    prediction_df[var] = values

# Add the gridmet_date column
prediction_df['gridmet_date'] = pd.to_datetime(latest_time)

# Save for enrichment
prediction_df.to_csv("./data/test/idaho_gridmet_latest.csv", index=False)
print("Idaho gridMET dataset with date ready for enrichment!")