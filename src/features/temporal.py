"""Temporal feature helpers shared by training and inference."""

import numpy as np
import pandas as pd


def month_to_season_num(month) -> float:
    """Map month (1-12) to season index: winter=0, spring=1, summer=2, fall=3."""
    if pd.isna(month):
        return np.nan
    month = int(month)
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    if month in (9, 10, 11):
        return 3
    return np.nan


def add_season_column(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """Add season_num column derived from month."""
    if month_col not in df.columns:
        return df
    df = df.copy()
    df["season_num"] = df[month_col].apply(month_to_season_num)
    return df
