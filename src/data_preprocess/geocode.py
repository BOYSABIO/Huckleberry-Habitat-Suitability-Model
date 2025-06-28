"""
Geocoding module for huckleberry habitat prediction.

This module handles geocoding of location data using multiple strategies
including traditional geocoding and LLM-assisted landmark extraction.
"""

import pandas as pd
import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import ollama
from tqdm import tqdm
from datetime import datetime
import os
from typing import Optional, Tuple, Dict, Any
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class Geocoder:
    """
    Handles geocoding of location data with multiple fallback strategies.
    """
    def __init__(self, use_llm: bool = True, llm_model: str = 'llama3:8b'):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.geocoder = self._init_geocoder()
        self._setup_logging()

    def _setup_logging(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'geocoding_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Geocoding log started: {log_file}")

    def _init_geocoder(self):
        geolocator = Nominatim(user_agent="huckleberry_habitat_prediction", timeout=10)
        return RateLimiter(geolocator.geocode, min_delay_seconds=1)

    def extract_landmark_llm(self, text: str) -> Optional[str]:
        if not self.use_llm:
            return None
        prompt = f"""
You are a geographic landmark parser.
From the following location description:
"{text}"
Extract the most specific, geocodable location information. Follow these rules in order:
1. If there's a distance-based location (e.g., "4 mi. E of Stevens Pass"), return the full distance-based location
2. If there's a specific natural feature (lake, mountain, river, canyon), return its name
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
            response = ollama.chat(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error in landmark extraction: {e}")
            return None

    def geocode_location(self, row: pd.Series) -> Tuple[float, float, bool]:
        if (pd.notnull(row['decimallatitude']) and pd.notnull(row['decimallongitude'])):
            return row['decimallatitude'], row['decimallongitude'], False
        attempts = []
        results = []
        fields = ['locality', 'municipality', 'county', 'stateprovince', 'countrycode']
        full_string = ", ".join(str(row[f]) for f in fields if pd.notnull(row[f]))
        if full_string.strip():
            attempts.append(("Full string", full_string))
        if pd.notnull(row['locality']) and pd.isnull(row['municipality']):
            locality_parts = [row['locality']]
            for f in ['county', 'stateprovince', 'countrycode']:
                if pd.notnull(row[f]):
                    locality_parts.append(str(row[f]))
            locality_string = ", ".join(locality_parts)
            if locality_string.strip():
                attempts.append(("Locality-based", locality_string))
        if pd.notnull(row['county']) and pd.notnull(row['stateprovince']):
            county_state = f"{row['county']}, {row['stateprovince']}"
            if pd.notnull(row['countrycode']):
                county_state += f", {row['countrycode']}"
            attempts.append(("County-State", county_state))
        if pd.notnull(row.get('verbatimlocality')):
            verbatim = str(row['verbatimlocality'])
            if (verbatim.lower() not in ['no verbatim locality recorded', 'nan', 'none', ''] and len(verbatim.strip()) > 5):
                attempts.append(("Verbatim locality", verbatim))
                verbatim_parts = [verbatim]
                for f in ['county', 'stateprovince', 'countrycode']:
                    if pd.notnull(row[f]):
                        verbatim_parts.append(str(row[f]))
                verbatim_with_context = ", ".join(verbatim_parts)
                attempts.append(("Verbatim with context", verbatim_with_context))
        for attempt_type, location_str in attempts:
            logger.info(f"Trying {attempt_type}: {location_str}")
            try:
                location = self.geocoder(location_str)
                if location:
                    logger.info(f"Success: {location.latitude}, {location.longitude}")
                    return location.latitude, location.longitude, False
            except Exception as e:
                logger.error(f"Error in {attempt_type} attempt: {e}")
                results.append((attempt_type, str(e)))
        if self.use_llm:
            locality_to_try = None
            if pd.notnull(row['locality']):
                locality_str = str(row['locality']).strip()
                if (locality_str.lower() not in ['no specific locality recorded', 'no locality recorded', 'nan', 'none', ''] and len(locality_str) > 5):
                    locality_to_try = locality_str
            if not locality_to_try and pd.notnull(row.get('verbatimlocality')):
                verbatim_str = str(row['verbatimlocality']).strip()
                if (verbatim_str.lower() not in ['no verbatim locality recorded', 'no locality recorded', 'nan', 'none', ''] and len(verbatim_str) > 5):
                    locality_to_try = verbatim_str
            if locality_to_try:
                landmark = self.extract_landmark_llm(locality_to_try)
                if landmark:
                    logger.info(f"Extracted landmark: {landmark}")
                    try:
                        location = self.geocoder(landmark)
                        if location:
                            logger.info(f"LLM Success: {location.latitude}, {location.longitude}")
                            return location.latitude, location.longitude, True
                    except Exception as e:
                        logger.error(f"LLM landmark error: {e}")
                    fallback_parts = [landmark]
                    for f in ['county', 'stateprovince', 'countrycode']:
                        if pd.notnull(row[f]):
                            fallback_parts.append(str(row[f]))
                    fallback_string = ", ".join(fallback_parts)
                    try:
                        location = self.geocoder(fallback_string)
                        if location:
                            logger.info(f"LLM Success with context: {location.latitude}, {location.longitude}")
                            return location.latitude, location.longitude, True
                    except Exception as e:
                        logger.error(f"LLM context error: {e}")
        if results:
            logger.warning(f"All geocoding attempts failed for row {row.name}")
            for attempt_type, error in results:
                logger.warning(f"- {attempt_type}: {error}")
        return None, None, False

    def geocode_dataset(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        logger.info(f"Starting geocoding of {len(df)} records")
        result_df = df.copy()
        if 'decimallatitude' not in result_df.columns:
            result_df['decimallatitude'] = None
        if 'decimallongitude' not in result_df.columns:
            result_df['decimallongitude'] = None
        result_df['used_llm'] = False
        total_batches = (len(result_df) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(result_df))
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (records {start_idx}-{end_idx})")
            batch_df = result_df.iloc[start_idx:end_idx]
            for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_idx + 1}"):
                lat, lon, used_llm = self.geocode_location(row)
                if lat is not None and lon is not None:
                    result_df.at[idx, 'decimallatitude'] = lat
                    result_df.at[idx, 'decimallongitude'] = lon
                    result_df.at[idx, 'used_llm'] = used_llm
                time.sleep(0.1)
        successful_geocoding = result_df['decimallatitude'].notna().sum()
        llm_used = result_df['used_llm'].sum()
        logger.info(f"Geocoding complete:")
        logger.info(f"  - Total records: {len(result_df)}")
        logger.info(f"  - Successfully geocoded: {successful_geocoding}")
        logger.info(f"  - Used LLM: {llm_used}")
        logger.info(f"  - Success rate: {successful_geocoding/len(result_df)*100:.1f}%")
        return result_df

    def get_geocoding_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_records = len(df)
        successful_geocoding = df['decimallatitude'].notna().sum()
        llm_used = df.get('used_llm', pd.Series([False] * len(df))).sum()
        summary = {
            'total_records': total_records,
            'successful_geocoding': successful_geocoding,
            'failed_geocoding': total_records - successful_geocoding,
            'success_rate': successful_geocoding / total_records * 100,
            'llm_used_count': llm_used,
            'llm_usage_rate': llm_used / total_records * 100
        }
        return summary

def load_manual_geocodes(path='src/data_preprocess/manual_geocodes.json') -> Dict[str, Dict[str, float]]:
    """
    Load manual geocode dictionary from a JSON file.
    Args:
        path: Path to the JSON file
    Returns:
        Dictionary mapping locality/municipality to lat/lon
    """
    with open(path, 'r') as f:
        return json.load(f)

def apply_manual_geocodes(df: pd.DataFrame, manual_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Apply manual geocodes to rows missing lat/lon but with locality/municipality.
    Args:
        df: DataFrame to update
        manual_dict: Dictionary of manual geocodes
    Returns:
        Updated DataFrame
    """
    for idx, row in df[df['decimallatitude'].isnull() | df['decimallongitude'].isnull()].iterrows():
        key = row.get('locality') or row.get('municipality')
        if key and key in manual_dict:
            df.at[idx, 'decimallatitude'] = manual_dict[key]['lat']
            df.at[idx, 'decimallongitude'] = manual_dict[key]['lon']
            logger.info(f"Applied manual geocode for {key} at index {idx}")
    return df 