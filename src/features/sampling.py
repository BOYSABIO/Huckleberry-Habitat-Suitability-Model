"""Training label sampling (pseudo-absences)."""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.features.temporal import add_season_column

logger = logging.getLogger(__name__)


def create_pseudo_absences(
    df: pd.DataFrame,
    ratio: float = 3.0,
    buffer_km: float = 5.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create pseudo-absence points around real occurrences.

    Args:
        df: Occurrence records with coordinates and datetime columns.
        ratio: Pseudo-absence to occurrence ratio.
        buffer_km: Minimum distance from real occurrences in km.
        random_seed: RNG seed for reproducibility.
    """
    logger.info("Creating pseudo-absences with ratio %s:1", ratio)

    np.random.seed(random_seed)
    df_copy = df.copy()
    df_copy["occurrence"] = 1

    coords_rad = np.radians(df_copy[["decimalLatitude", "decimalLongitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    buffer_rad = buffer_km / 6371

    num_absences = int(df_copy.shape[0] * ratio)
    pseudo_points = []

    lat_range = (df_copy["decimalLatitude"].min(), df_copy["decimalLatitude"].max())
    lon_range = (df_copy["decimalLongitude"].min(), df_copy["decimalLongitude"].max())
    date_range = (
        pd.to_datetime(df_copy["datetime"]).min(),
        pd.to_datetime(df_copy["datetime"]).max(),
    )

    gridmet_columns = [col for col in df_copy.columns if col != "occurrence"]

    attempts = 0
    max_attempts = num_absences * 20
    while len(pseudo_points) < num_absences and attempts < max_attempts:
        lat = np.random.uniform(*lat_range)
        lon = np.random.uniform(*lon_range)
        coord_rad = np.radians([[lat, lon]])

        dist, _ = tree.query(coord_rad, k=1)
        if dist[0][0] >= buffer_rad:
            random_date = date_range[0] + timedelta(
                days=np.random.randint(0, (date_range[1] - date_range[0]).days + 1)
            )
            row = {
                "decimalLatitude": lat,
                "decimalLongitude": lon,
                "datetime": random_date.strftime("%Y-%m-%d"),
                "year": random_date.year,
                "month": random_date.month,
                "day": random_date.day,
                "occurrence": 0,
            }
            for col in gridmet_columns:
                if col not in row:
                    row[col] = np.nan
            pseudo_points.append(row)
        attempts += 1

    pseudo_df = pd.DataFrame(pseudo_points)[df_copy.columns]
    combined_df = (
        pd.concat([df_copy, pseudo_df], ignore_index=True)
        .pipe(add_season_column)
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )

    logger.info(
        "Created %s pseudo-absences, total dataset: %s records",
        len(pseudo_points),
        len(combined_df),
    )
    return combined_df
