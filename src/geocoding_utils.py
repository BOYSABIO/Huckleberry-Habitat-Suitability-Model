from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
import pandas as pd
from tqdm import tqdm
import json
import os
from geopy.extra.rate_limiter import RateLimiter
import ollama


def init_geocoder():
    """Initialize the geocoder with rate limiting."""
    geolocator = Nominatim(user_agent="geoapi_huckleberries_llm", timeout=30)
    return RateLimiter(geolocator.geocode, min_delay_seconds=1)


def extract_landmark_ollama(text, model='llama3:8b'):
    """Extract the most specific natural feature from a location description."""
    prompt = f"""
You are a geographic landmark parser.

From the following location description:

"{text}"

Extract the most specific, geocodable **natural feature** such as a lake, mountain, river, or canyon — preferably **a lake if mentioned**. Return just the name of the place that would most help identify the location on a map.

If multiple are mentioned, choose the **most specific or closest landmark** mentioned.

Return just the name of the place. No explanations.
"""
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error in landmark extraction: {e}")
        return None


def geocode_with_llm_fallback(row, geocode):
    """Geocode a row using multiple strategies with LLM fallback."""
    if pd.isnull(row['decimalLatitude']) or pd.isnull(row['decimalLongitude']):
        attempts = []
        results = []

        # Full location string attempt
        fields = ['locality', 'municipality', 'county', 'stateProvince', 'countryCode']
        full_string = ", ".join(str(row[f]) for f in fields if pd.notnull(row[f]))
        attempts.append(("Full string", full_string))

        # Landmark extraction fallback
        if pd.notnull(row['locality']):
            landmark = extract_landmark_ollama(row['locality'])
            if landmark:
                print(f"Extracted landmark: {landmark}")
                fallback_string = ", ".join([landmark] + 
                    [str(row[f]) for f in ['county', 'stateProvince', 'countryCode'] 
                     if pd.notnull(row[f])])
                attempts.append(("Landmark", fallback_string))

        # Try all attempts
        for attempt_type, location_str in attempts:
            print(f"Trying {attempt_type}: {location_str}")
            try:
                location = geocode(location_str)
                if location:
                    print(f"Success: {location.latitude}, {location.longitude}")
                    return pd.Series([location.latitude, location.longitude])
            except Exception as e:
                print(f"Error in {attempt_type} attempt: {e}")
                results.append((attempt_type, str(e)))

        # If all attempts failed, log the failures
        if results:
            print(f"All attempts failed for row {row.name}:")
            for attempt_type, error in results:
                print(f"- {attempt_type}: {error}")

    return pd.Series([row['decimalLatitude'], row['decimalLongitude']])


def create_address_string(row, include_locality=True, include_municipality=True):
    """Combine address components into a single string with better structure."""
    address_parts = []
    
    # Start with the most specific location if requested
    if include_locality and pd.notna(row.get('locality')):
        # Clean up the locality - remove distance references
        locality = str(row['locality'])
        # Remove common distance patterns
        locality = ' '.join([word for word in locality.split() 
                           if not any(unit in word.lower() 
                                    for unit in ['mile', 'miles', 'km', 'kilometer'])])
        if locality.strip():  # Only add if we have something left
            address_parts.append(locality)
    
    # Add municipality if requested and available
    if include_municipality and pd.notna(row.get('municipality')):
        address_parts.append(str(row['municipality']))
    
    # Add county if available
    if pd.notna(row.get('county')):
        county = str(row['county'])
        # Remove "Co." or "County" if present
        county = county.replace(' Co.', '').replace(' County', '')
        address_parts.append(f"{county} County")
    
    # Add state if available
    if pd.notna(row.get('stateProvince')):
        state = str(row['stateProvince'])
        # Convert state abbreviations to full names
        state_map = {
            'MT': 'Montana', 'WY': 'Wyoming', 'ID': 'Idaho',
            'WA': 'Washington', 'OR': 'Oregon', 'CA': 'California',
            'NV': 'Nevada', 'AZ': 'Arizona', 'UT': 'Utah',
            'CO': 'Colorado', 'NM': 'New Mexico', 'ND': 'North Dakota',
            'SD': 'South Dakota', 'NE': 'Nebraska', 'KS': 'Kansas',
            'OK': 'Oklahoma', 'TX': 'Texas', 'MN': 'Minnesota',
            'IA': 'Iowa', 'MO': 'Missouri', 'AR': 'Arkansas',
            'LA': 'Louisiana', 'WI': 'Wisconsin', 'IL': 'Illinois',
            'MS': 'Mississippi', 'MI': 'Michigan', 'IN': 'Indiana',
            'OH': 'Ohio', 'KY': 'Kentucky', 'TN': 'Tennessee',
            'AL': 'Alabama', 'GA': 'Georgia', 'FL': 'Florida',
            'SC': 'South Carolina', 'NC': 'North Carolina', 'VA': 'Virginia',
            'WV': 'West Virginia', 'MD': 'Maryland', 'DE': 'Delaware',
            'PA': 'Pennsylvania', 'NJ': 'New Jersey', 'NY': 'New York',
            'CT': 'Connecticut', 'RI': 'Rhode Island', 'MA': 'Massachusetts',
            'NH': 'New Hampshire', 'VT': 'Vermont', 'ME': 'Maine',
            'AK': 'Alaska', 'HI': 'Hawaii'
        }
        state = state_map.get(state.upper(), state)
        address_parts.append(state)
    
    # Always add USA at the end
    address_parts.append("USA")
    
    # Join all parts with commas
    return ', '.join(address_parts)


def geocode_address(geolocator, row, max_retries=5):
    """Geocode an address with retry logic and exponential backoff."""
    # Try different combinations of address components
    address_combinations = [
        (True, True),   # With locality and municipality
        (False, True),  # Without locality, with municipality
        (False, False)  # Without locality and municipality
    ]
    
    for include_locality, include_municipality in address_combinations:
        address = create_address_string(row, include_locality, include_municipality)
        print(f"\nTrying address: {address}")
        
        for attempt in range(max_retries):
            try:
                location = geolocator.geocode(address)
                if location:
                    print(f"Successfully geocoded: {location.address}")
                    return location.latitude, location.longitude
                print("No results found for this address")
                break  # If no result, try next combination
            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                if attempt == max_retries - 1:
                    print(f"Failed to geocode address after {max_retries} attempts")
                    break
                wait_time = (2 ** attempt) * 2
                print(f"Attempt {attempt + 1} failed, waiting {wait_time} seconds...")
                time.sleep(wait_time)
    
    return None


def save_checkpoint(coordinates, current_index, output_file='geocoding_checkpoint.json'):
    """Save the current progress to a checkpoint file."""
    # Convert coordinates to a serializable format
    serializable_coords = []
    for coord in coordinates:
        if coord is None:
            serializable_coords.append(None)
        else:
            serializable_coords.append([float(coord[0]), float(coord[1])])
    
    checkpoint = {
        'coordinates': serializable_coords,
        'current_index': int(current_index)  # Ensure index is an integer
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(checkpoint, f)
        print(f"Checkpoint saved to: {os.path.abspath(output_file)}")
        print(f"Saved {len(coordinates)} coordinates up to index {current_index}")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")


def save_results_to_csv(missing_coords, coordinates, output_file='geocoding_results.csv'):
    """Save the geocoding results to a CSV file."""
    try:
        # Create a DataFrame with the results
        results_df = pd.DataFrame({
            'index': missing_coords.index,
            'locality': missing_coords['locality'],
            'municipality': missing_coords['municipality'],
            'county': missing_coords['county'],
            'stateProvince': missing_coords['stateProvince'],
            'latitude': [coord[0] if coord else None for coord in coordinates],
            'longitude': [coord[1] if coord else None for coord in coordinates]
        })
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Error saving results to CSV: {str(e)}")


def load_checkpoint(output_file='geocoding_checkpoint.json'):
    """Load progress from a checkpoint file if it exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint with {len(checkpoint['coordinates'])} coordinates")
            print(f"Last processed index: {checkpoint['current_index']}")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    return None


def geocode_missing_coordinates(df, checkpoint_file='geocoding_checkpoint.json', results_file='geocoding_results.csv'):
    """
    Geocode addresses for rows missing coordinates using LLM-based landmark extraction.
    
    Args:
        df (pandas.DataFrame): DataFrame containing location information
        checkpoint_file (str): Path to the checkpoint file
        results_file (str): Path to save the results CSV
        
    Returns:
        pandas.DataFrame: DataFrame with new latitude and longitude columns
    """
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    print(f"Checkpoint file will be saved to: {os.path.abspath(checkpoint_file)}")
    print(f"Results file will be saved to: {os.path.abspath(results_file)}")
    
    # Initialize geocoder
    geocode = init_geocoder()
    
    # Get rows without coordinates
    missing_coords = df[df['hasCoordinate'] == False].copy()
    print(f"Found {len(missing_coords)} rows without coordinates")
    
    # Try to load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        print(f"Found checkpoint at index {checkpoint['current_index']}")
        coordinates = checkpoint['coordinates']
        start_idx = checkpoint['current_index'] + 1  # Start from the next row
        print(f"Resuming from row {start_idx}")
    else:
        print("No checkpoint found, starting from beginning")
        coordinates = []
        start_idx = 0
    
    # Geocode addresses (with progress bar)
    print("Geocoding addresses...")
    success_count = 0
    error_count = 0
    
    try:
        for idx, row in tqdm(missing_coords.iloc[start_idx:].iterrows(), total=len(missing_coords) - start_idx):
            try:
                print(f"\nProcessing row {idx}:")
                new_coords = geocode_with_llm_fallback(row, geocode)
                
                # Update the coordinates
                missing_coords.loc[idx, 'new_latitude'] = new_coords[0]
                missing_coords.loc[idx, 'new_longitude'] = new_coords[1]
                
                # Update counts
                if pd.notnull(new_coords[0]) and pd.notnull(new_coords[1]):
                    success_count += 1
                else:
                    error_count += 1
                
                # Save checkpoint and results every 10 rows
                if (idx + 1) % 10 == 0:
                    print(f"\nSaving checkpoint at row {idx + 1}...")
                    save_checkpoint(coordinates, idx, checkpoint_file)
                    save_results_to_csv(missing_coords.iloc[:idx+1], coordinates, results_file)
                    print(f"\nProgress update:")
                    print(f"Processed {idx + 1} rows")
                    print(f"Successful: {success_count}")
                    print(f"Failed: {error_count}")
                
                # Add a small delay between rows
                time.sleep(1)
                
            except Exception as e:
                print(f"\nERROR processing row {idx}:")
                print(f"Error message: {str(e)}")
                error_count += 1
                missing_coords.loc[idx, 'new_latitude'] = None
                missing_coords.loc[idx, 'new_longitude'] = None
                time.sleep(2)  # Longer delay after an error
        
        # Save final results
        print("\nSaving final results...")
        save_results_to_csv(missing_coords, coordinates, results_file)
        
        # Print final statistics
        print("\nFinal statistics:")
        print(f"Total rows processed: {len(missing_coords)}")
        print(f"Successfully geocoded: {success_count}")
        print(f"Failed to geocode: {error_count}")
        
        # Remove checkpoint file if we completed successfully
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Removed checkpoint file: {checkpoint_file}")
        
        return missing_coords
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        print("Progress has been saved. You can resume from the last checkpoint.")
        # Save results even if there's an error
        save_results_to_csv(missing_coords.iloc[:len(coordinates)], coordinates, results_file)
        return None