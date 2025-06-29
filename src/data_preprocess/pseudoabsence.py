import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from datetime import timedelta


def generate_pseudo_absences(df_occurrences, ratio=3, buffer_km=5, random_seed=42):
    """
    Generate pseudo-absence points for habitat modeling.
    - ratio: number of pseudo-absences per occurrence
    - buffer_km: minimum distance from any occurrence (km)
    - random_seed: for reproducibility
    Returns a combined DataFrame with both occurrences and pseudo-absences.
    """
    np.random.seed(random_seed)
    df = df_occurrences.copy()
    df['occurrence'] = 1  # Ensure all are marked as occurrences

    coords_rad = np.radians(df[["decimalLatitude", "decimalLongitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    buffer_rad = buffer_km / 6371  # Earth's radius in km

    num_absences = int(df.shape[0] * ratio)
    pseudo_points = []

    lat_range = (df["decimalLatitude"].min(), df["decimalLatitude"].max())
    lon_range = (df["decimalLongitude"].min(), df["decimalLongitude"].max())
    date_range = (pd.to_datetime(df["datetime"]).min(), pd.to_datetime(df["datetime"]).max())

    # Get list of columns to preserve structure
    gridmet_columns = [col for col in df.columns if col != "occurrence"]

    attempts = 0
    max_attempts = num_absences * 20
    while len(pseudo_points) < num_absences and attempts < max_attempts:
        lat = np.random.uniform(*lat_range)
        lon = np.random.uniform(*lon_range)
        coord_rad = np.radians([[lat, lon]])

        dist, _ = tree.query(coord_rad, k=1)
        if dist[0][0] >= buffer_rad:
            random_date = date_range[0] + timedelta(days=np.random.randint(0, (date_range[1] - date_range[0]).days + 1))
            row = {
                "decimalLatitude": lat,
                "decimalLongitude": lon,
                "datetime": random_date.strftime("%Y-%m-%d"),
                "occurrence": 0
            }
            # Fill rest of columns with NaN to match structure
            for col in gridmet_columns:
                if col not in row:
                    row[col] = np.nan
            pseudo_points.append(row)
        attempts += 1

    pseudo_df = pd.DataFrame(pseudo_points)[df.columns]  # reorder to match
    combined = pd.concat([df, pseudo_df], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return combined 