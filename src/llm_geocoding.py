import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import ollama
from tqdm import tqdm

def init_geocoder():
    """Initialize the geocoder with rate limiting."""
    geolocator = Nominatim(user_agent="geoapi_huckleberries_llm", timeout=10)
    return RateLimiter(geolocator.geocode, min_delay_seconds=1)

def extract_landmark_ollama(text, model='llama3:8b'):
    """Extract the most specific natural feature from a location description."""
    prompt = f"""
You are a geographic landmark parser.

From the following location description:

"{text}"

Extract the most specific, geocodable **natural feature** such as a lake, 
mountain, river, or canyon — preferably **a lake if mentioned**. Return just 
the name of the place that would most help identify the location on a map.

If multiple are mentioned, choose the **most specific or closest landmark** 
mentioned.

If it is a **natural feature**, make sure you have the name of the natural feature as well.
If not then choose the next best option.
For example, if the location description is "Chelan Co., Mazama: Along rocky shore of Lake ca. 3 miles S of route 20 in Rainey Pass, Washington, US",
You would not pass "lake" and instead return "Rainey Pass" because we don't have the name of the lake.

Return just the name of the place. No explanations.
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error in landmark extraction: {e}")
        return None

def geocode_with_llm_fallback(row, geocode):
    """Geocode a row using multiple strategies with LLM fallback."""
    if pd.isnull(row['decimalLatitude']) or pd.isnull(row['decimalLongitude']):
        # First try regular geocoding without LLM
        fields = ['locality', 'municipality', 'county', 'stateProvince', 
                 'countryCode']
        full_string = ", ".join(str(row[f]) for f in fields if pd.notnull(row[f]))
        
        print(f"Trying regular geocoding: {full_string}")
        try:
            location = geocode(full_string)
            if location:
                print(f"Success with regular geocoding: {location.latitude}, {location.longitude}")
                return pd.Series([location.latitude, location.longitude, False])
        except Exception as e:
            print(f"Regular geocoding failed: {e}")

        # If regular geocoding fails, try LLM approach
        if pd.notnull(row['locality']):
            landmark = extract_landmark_ollama(row['locality'])
            if landmark:
                print(f"Extracted landmark: {landmark}")
                fallback_parts = [landmark]
                for f in ['county', 'stateProvince', 'countryCode']:
                    if pd.notnull(row[f]):
                        fallback_parts.append(str(row[f]))
                fallback_string = ", ".join(fallback_parts)
                
                print(f"Trying LLM-assisted geocoding: {fallback_string}")
                try:
                    location = geocode(fallback_string)
                    if location:
                        print(f"Success with LLM: {location.latitude}, {location.longitude}")
                        return pd.Series([location.latitude, location.longitude, True])
                except Exception as e:
                    print(f"LLM-assisted geocoding failed: {e}")

    return pd.Series([row['decimalLatitude'], row['decimalLongitude'], False])

def process_dataset(df):
    """
    Process the entire dataset to update missing coordinates using LLM assistance.
    
    Args:
        df (pandas.DataFrame): DataFrame containing location information
        
    Returns:
        tuple: (updated_df, stats) where stats is a dict containing update statistics
    """
    # Initialize geocoder
    geocode = init_geocoder()
    
    # Make a copy of the dataframe
    df_updated = df.copy()
    
    # Track statistics
    stats = {
        'total_rows': len(df),
        'missing_coords': 0,
        'successfully_updated': 0,
        'failed_updates': 0,
        'llm_used_count': 0
    }
    
    # Count initial missing coordinates
    missing_mask = df['decimalLatitude'].isna() | df['decimalLongitude'].isna()
    stats['missing_coords'] = missing_mask.sum()
    
    print(f"Processing {stats['missing_coords']} rows with missing coordinates...")
    
    # Process rows with missing coordinates
    for idx in tqdm(df[missing_mask].index):
        original_coords = (df.loc[idx, 'decimalLatitude'], 
                         df.loc[idx, 'decimalLongitude'])
        new_coords = geocode_with_llm_fallback(df.loc[idx], geocode)
        
        # Update coordinates and LLM usage
        df_updated.loc[idx, 'decimalLatitude'] = new_coords[0]
        df_updated.loc[idx, 'decimalLongitude'] = new_coords[1]
        df_updated.loc[idx, 'used_llm'] = new_coords[2]
        
        # Update statistics
        if pd.notna(new_coords[0]) and pd.notna(new_coords[1]):
            stats['successfully_updated'] += 1
            if new_coords[2]:  # If LLM was used
                stats['llm_used_count'] += 1
        else:
            stats['failed_updates'] += 1
    
    return df_updated, stats

def main():
    """Main function to demonstrate usage."""
    # Example usage
    # df = pd.read_csv('your_data.csv')
    # df_updated, stats = process_dataset(df)
    
    # Print statistics
    print("\nGeocoding Statistics:")
    print(f"Total rows processed: {stats['total_rows']}")
    print(f"Rows with missing coordinates: {stats['missing_coords']}")
    print(f"Successfully updated: {stats['successfully_updated']}")
    print(f"Failed updates: {stats['failed_updates']}")
    success_rate = (stats['successfully_updated'] / stats['missing_coords'] * 100)
    print(f"Success rate: {success_rate:.2f}%")

if __name__ == "__main__":
    main() 