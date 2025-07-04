import pandas as pd
import requests
import os
import time
from tqdm import tqdm

# ---------------- CONFIG ----------------
INPUT_CSV = "./data/enriched/HB_PSEUDO_clean_elevation.csv"
OUTPUT_CSV = "./data/enriched/HB_PSEUDO_clean_elevation_soil.csv"
SOILGRID_API = "https://rest.isric.org/soilgrids/v2.0/properties/query"
DEPTH = "15-30cm"
PROPERTY = "phh2o"
# ---------------------------------------

def get_soil_ph(lat, lon, retries=5, base_delay=5):
    """Fetch soil pH and retry on 429 with exponential backoff."""
    url = (
        f"{SOILGRID_API}?lat={lat}&lon={lon}"
        f"&property={PROPERTY}&depth={DEPTH}&value=mean"
    )

    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 429:
                wait = base_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()
            layers = data.get("properties", {}).get("layers", [])
            for layer in layers:
                if layer["name"] == PROPERTY:
                    values = layer["depths"][0]["values"]
                    raw_ph = values.get("mean")
                    if raw_ph is not None:
                        return round(raw_ph / 10.0, 2)  # Convert to pH scale
            return None

        except Exception as e:
            print(f"Error at ({lat}, {lon}): {e}")
            time.sleep(base_delay * (2 ** attempt))
    
    return None  # failed after retries

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)

    if os.path.exists(OUTPUT_CSV):
        enriched = pd.read_csv(OUTPUT_CSV)
        enriched_ids = set(enriched.index)
        append_mode = True
    else:
        enriched_ids = set()
        append_mode = False

    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f:
        for i, row in tqdm(df.iterrows(), total=len(df), desc="🔬 Enriching Soil pH"):
            if i in enriched_ids:
                continue

            lat = row["decimalLatitude"]
            lon = row["decimalLongitude"]

            ph = get_soil_ph(lat, lon)
            row_dict = row.to_dict()
            row_dict["soil_ph"] = ph

            # Write header only if not already done
            pd.DataFrame([row_dict]).to_csv(f, index=False, header=not append_mode)
            append_mode = True  # after first row, disable header writing

if __name__ == "__main__":
    main()
