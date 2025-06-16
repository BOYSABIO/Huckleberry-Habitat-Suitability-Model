import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import ollama
from tqdm import tqdm
import logging
from datetime import datetime
import os

def setup_logging():
    """Set up logging to file only, with no console output."""
    # Create logs directory in root if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'geocoding_{timestamp}.log')
    
    # Configure logging to file only
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file)],
        force=True
    )
    
    # Disable console output
    logging.getLogger().handlers = [h for h in logging.getLogger().handlers 
                                  if isinstance(h, logging.FileHandler)]
    
    return log_file

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

Extract the most specific, geocodable location information. Follow these rules in order:

1. If there's a distance-based location (e.g., "4 mi. E of Stevens Pass"), 
   return the full distance-based location
2. If there's a specific natural feature (lake, mountain, river, canyon), 
   return its name
3. If there's a National Forest or Park mentioned, return its full name
4. If there's a mountain range mentioned, return its name
5. If there's a regional district mentioned, return its name
6. If multiple landmarks are mentioned, choose the most specific one

For example:
- "4 mi. E of Stevens Pass" -> "4 mi. E of Stevens Pass"
- "Spanish Basin, Madison Range" -> "Spanish Basin"
- "Near Lake Bootahnie, Marble Mountains" -> "Lake Bootahnie"
- "Nez Perce Natl. For." -> "Nez Perce National Forest"
- "Columbia-Shuswap Regional District" -> "Columbia-Shuswap Regional District"

Return just the name of the place. No explanations.
"""
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error in landmark extraction: {e}")
        return None

def geocode_with_llm_fallback(row, geocode):
    """Geocode a row using multiple strategies with LLM fallback."""
    if pd.isnull(row['decimalLatitude']) or pd.isnull(row['decimalLongitude']):
        attempts = []
        results = []

        # Strategy 1: Full location string with all available fields
        fields = ['locality', 'municipality', 'county', 'stateProvince', 'countryCode']
        full_string = ", ".join(str(row[f]) for f in fields if pd.notnull(row[f]))
        if full_string.strip():
            attempts.append(("Full string", full_string))

        # Strategy 2: Try with locality and county/state if municipality is missing
        if pd.notnull(row['locality']) and pd.isnull(row['municipality']):
            locality_parts = [row['locality']]
            for f in ['county', 'stateProvince', 'countryCode']:
                if pd.notnull(row[f]):
                    locality_parts.append(str(row[f]))
            locality_string = ", ".join(locality_parts)
            if locality_string.strip():
                attempts.append(("Locality-based", locality_string))

        # Strategy 3: Try with just county and state if we have them
        if pd.notnull(row['county']) and pd.notnull(row['stateProvince']):
            county_state = f"{row['county']}, {row['stateProvince']}"
            if pd.notnull(row['countryCode']):
                county_state += f", {row['countryCode']}"
            attempts.append(("County-State", county_state))

        # Try all non-LLM attempts first
        for attempt_type, location_str in attempts:
            logging.info(f"Trying {attempt_type}: {location_str}")
            try:
                location = geocode(location_str)
                if location:
                    logging.info(f"Success: {location.latitude}, {location.longitude}")
                    return pd.Series([location.latitude, location.longitude, False])
            except Exception as e:
                logging.error(f"Error in {attempt_type} attempt: {e}")
                results.append((attempt_type, str(e)))

        # Strategy 4: Enhanced LLM landmark extraction as last resort
        if pd.notnull(row['locality']):
            landmark = extract_landmark_ollama(row['locality'])
            if landmark:
                logging.info(f"Extracted landmark: {landmark}")
                # Try the landmark alone first
                try:
                    location = geocode(landmark)
                    if location:
                        msg = f"LLM Success (landmark only): {location.latitude}, "
                        msg += f"{location.longitude}"
                        logging.info(msg)
                        return pd.Series([location.latitude, location.longitude, True])
                except Exception as e:
                    logging.error(f"Error in LLM landmark-only attempt: {e}")
                
                # If landmark alone fails, try with additional context
                fallback_parts = [landmark]
                for f in ['county', 'stateProvince', 'countryCode']:
                    if pd.notnull(row[f]):
                        fallback_parts.append(str(row[f]))
                fallback_string = ", ".join(fallback_parts)
                
                try:
                    location = geocode(fallback_string)
                    if location:
                        msg = f"LLM Success (with context): {location.latitude}, "
                        msg += f"{location.longitude}"
                        logging.info(msg)
                        return pd.Series([location.latitude, location.longitude, True])
                except Exception as e:
                    logging.error(f"Error in LLM attempt with context: {e}")
                    results.append(("LLM", str(e)))

        # If all attempts failed, log the failures
        if results:
            logging.warning(f"All attempts failed for row {row.name}:")
            for attempt_type, error in results:
                logging.warning(f"- {attempt_type}: {error}")

    return pd.Series([row['decimalLatitude'], row['decimalLongitude'], False])

def process_dataset(df):
    """
    Process the entire dataset to update missing coordinates using LLM assistance.
    
    Args:
        df (pandas.DataFrame): DataFrame containing location information
        
    Returns:
        tuple: (updated_df, stats) where stats is a dict containing update statistics
    """
    # Set up logging
    log_file = setup_logging()
    
    # Initialize geocoder
    geocode = init_geocoder()
    
    # Make a copy of the dataframe
    df_updated = df.copy()
    
    # Add column to track LLM usage
    df_updated['used_llm'] = False
    
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
    
    # Process rows with missing coordinates
    with tqdm(df[missing_mask].index, desc="Geocoding progress", leave=True) as pbar:
        for idx in pbar:
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
    
    # Log final statistics
    logging.info("\nGeocoding Statistics:")
    logging.info(f"Total rows processed: {stats['total_rows']}")
    logging.info(f"Rows with missing coordinates: {stats['missing_coords']}")
    logging.info(f"Successfully updated: {stats['successfully_updated']}")
    logging.info(f"Failed updates: {stats['failed_updates']}")
    logging.info(f"Rows where LLM was used: {stats['llm_used_count']}")
    success_rate = (stats['successfully_updated'] / stats['missing_coords'] * 100)
    logging.info(f"Success rate: {success_rate:.2f}%")
    
    # Print final statistics to notebook
    print("\nGeocoding Statistics:")
    print(f"Total rows processed: {stats['total_rows']}")
    print(f"Rows with missing coordinates: {stats['missing_coords']}")
    print(f"Successfully updated: {stats['successfully_updated']}")
    print(f"Failed updates: {stats['failed_updates']}")
    print(f"Rows where LLM was used: {stats['llm_used_count']}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"\nDetailed logs saved to: {log_file}")
    
    return df_updated, stats 