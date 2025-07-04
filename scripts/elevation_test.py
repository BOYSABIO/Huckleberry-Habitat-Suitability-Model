import pandas as pd
import requests
import time
from tqdm import tqdm
import os

# ------------ CONFIG ------------
INPUT_CSV = "./data/test/idaho_gridmet_latest.csv"
OUTPUT_CSV = "./data/test/idaho_gridmet_elevation.csv"
API_URL = "https://api.open-elevation.com/api/v1/lookup"
REQUEST_DELAY = 1  # seconds between API calls to be polite
# --------------------------------

def get_elevation(lat, lon):
    """Query Open-Elevation API for elevation."""
    try:
        response = requests.get(API_URL, params={"locations": f"{lat},{lon}"})
        if response.status_code == 200:
            elevation = response.json()["results"][0]["elevation"]
            return elevation
        else:
            print(f"⚠️ API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Failed to get elevation for {lat}, {lon}: {e}")
        return None

def main():
    df = pd.read_csv(INPUT_CSV)
    
    # If elevation column doesn't exist, add it
    if "elevation" not in df.columns:
        df["elevation"] = pd.NA

    # Resume from existing file
    if os.path.exists(OUTPUT_CSV):
        enriched = pd.read_csv(OUTPUT_CSV)
        start_idx = len(enriched)
        print(f"Resuming from row {start_idx}")
    else:
        enriched = pd.DataFrame()
        start_idx = 0

    for i in tqdm(range(start_idx, len(df)), desc="Adding elevation"):
        row = df.iloc[i]

        if pd.notna(row.get("elevation", None)):
            enriched = pd.concat([enriched, row.to_frame().T], ignore_index=True)
            continue

        lat = row["decimalLatitude"]
        lon = row["decimalLongitude"]
        elevation = get_elevation(lat, lon)

        row["elevation"] = elevation
        enriched = pd.concat([enriched, row.to_frame().T], ignore_index=True)

        # Save progress
        enriched.to_csv(OUTPUT_CSV, index=False)

        time.sleep(REQUEST_DELAY)  # avoid hammering API

    print("Elevation enrichment complete.")

if __name__ == "__main__":
    main()