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
import requests
import re

logger = logging.getLogger(__name__)

class Geocoder:
    """
    Handles geocoding of location data with multiple fallback strategies.
    """
    def __init__(self, llm_model: str = "llama3:8b", 
                 llm_enabled: bool = True):
        self.llm_model = llm_model
        self.llm_enabled = llm_enabled
        self.geocoding_api = "https://nominatim.openstreetmap.org/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HuckleberryHabitat/1.0 (educational project)'
        })
        
        # Initialize geocoder
        geolocator = Nominatim(user_agent="huckleberry_habitat_prediction", timeout=10)
        self.geocoder = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        # Load manual geocodes
        self.manual_geocodes = self._load_manual_geocodes()

    def _load_manual_geocodes(self) -> Dict[str, Tuple[float, float]]:
        """Load manual geocodes from JSON file."""
        try:
            with open('src/data_preprocess/manual_geocodes.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Manual geocodes file not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading manual geocodes: {e}")
            return {}

    def _extract_landmark_llm(self, locality: str, county: str, 
                             state: str, country: str) -> Optional[str]:
        """Extract landmark using LLM."""
        if not self.llm_enabled:
            return None
            
        prompt = f"""Extract the most geocodable landmark from this location. Prioritize:
1. Lakes, rivers, streams, creeks
2. National Forests, National Parks, State Parks  
3. Mountains, peaks, ranges
4. Other named natural features

Location: {locality}, {county}, {state}, {country}

Return only the landmark name and only the one. Examples:
- "Upper Holland Lake" (not "Swan Range")
- "Echo Lake" (not "Rocky Mountains")
- "Flathead National Forest"
- "Yellowstone National Park"

Landmark:"""
        
        try:
            response = ollama.chat(
                model=self.llm_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            content = response['message']['content'].strip()
            
            # Clean up the response to extract just the landmark name
            # Remove any explanatory text and get the last line (the actual landmark)
            lines = content.split('\n')
            landmark = None
            
            # Look for the actual landmark in the response
            for line in lines:
                line = line.strip()
                # Skip empty lines and explanatory text
                if (line and 
                    not line.startswith('The most specific') and
                    not line.startswith('Answer:') and
                    not line.startswith('Return') and
                    not line.startswith('For example') and
                    not line.startswith('Based on') and
                    not line.startswith('Given') and
                    not line.startswith('Location:') and
                    not line.startswith('Prioritize') and
                    not line.startswith('1.') and
                    not line.startswith('2.') and
                    not line.startswith('3.') and
                    not line.startswith('4.') and
                    not line.startswith('-') and
                    not line.startswith('If no') and
                    len(line) > 2):
                    landmark = line
                    break
            
            # If no clean landmark found, try to extract from the last non-empty line
            if not landmark:
                for line in reversed(lines):
                    line = line.strip()
                    if line and len(line) > 2:
                        landmark = line
                        break
            
            return landmark
        except Exception as e:
            logger.error(f"Error in landmark extraction: {e}")
            return None

    def geocode_location(self, row: pd.Series) -> Tuple[Optional[float], Optional[float], bool]:
        if (pd.notnull(row['decimalLatitude']) and pd.notnull(row['decimalLongitude'])):
            return row['decimalLatitude'], row['decimalLongitude'], False
        
        attempts = []
        results = []
        
        # Strategy 1: Full string with all available fields (most context)
        fields = ['locality', 'municipality', 'county', 'stateProvince', 'countryCode']
        full_string = ", ".join(str(row[f]) for f in fields if pd.notnull(row[f]))
        if full_string.strip():
            attempts.append(("Full string", full_string))
        
        # Strategy 2: Most granular - specific locality/verbatim alone (if meaningful)
        # Check both verbatimLocality and locality as they contain the same type of data
        specific_locations = []
        
        # Check verbatimLocality
        if pd.notnull(row.get('verbatimLocality')):
            verbatim = str(row['verbatimLocality']).strip()
            if (verbatim.lower() not in ['no verbatim locality recorded', 'nan', 'none', ''] and 
                len(verbatim) > 5):
                specific_locations.append(("Verbatim locality only", verbatim))
        
        # Check locality
        if pd.notnull(row['locality']):
            locality_str = str(row['locality']).strip()
            if (locality_str.lower() not in ['no specific locality recorded', 'no locality recorded', 'nan', 'none', ''] and 
                len(locality_str) > 5):
                specific_locations.append(("Locality only", locality_str))
        
        # Add the specific location attempts
        attempts.extend(specific_locations)
        
        # Strategy 3: Specific location with minimal context (county + state)
        # Try verbatimLocality with context
        if pd.notnull(row.get('verbatimLocality')):
            verbatim = str(row['verbatimLocality']).strip()
            if (verbatim.lower() not in ['no verbatim locality recorded', 'nan', 'none', ''] and 
                len(verbatim) > 5):
                verbatim_parts = [verbatim]
                for f in ['county', 'stateProvince']:
                    if pd.notnull(row[f]):
                        verbatim_parts.append(str(row[f]))
                verbatim_with_context = ", ".join(verbatim_parts)
                attempts.append(("Verbatim with county/state", verbatim_with_context))
        
        # Try locality with context
        if pd.notnull(row['locality']):
            locality_parts = [row['locality']]
            for f in ['county', 'stateProvince']:
                if pd.notnull(row[f]):
                    locality_parts.append(str(row[f]))
            locality_string = ", ".join(locality_parts)
            if locality_string.strip():
                attempts.append(("Locality with county/state", locality_string))
        
        # Try all non-LLM attempts first
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
        
        # Strategy 4: LLM landmark extraction (before last resort)
        if self.llm_enabled:
            logger.info("LLM is enabled, attempting landmark extraction...")
            locality_to_try = None
            # Try locality first, then verbatimLocality if locality is not available
            if pd.notnull(row['locality']):
                locality_str = str(row['locality']).strip()
                if (locality_str.lower() not in ['no specific locality recorded', 'no locality recorded', 'nan', 'none', ''] and 
                    len(locality_str) > 5):
                    locality_to_try = locality_str
                    logger.info(f"Will try LLM with locality: '{locality_str}'")
            elif pd.notnull(row.get('verbatimLocality')):
                verbatim_str = str(row['verbatimLocality']).strip()
                if (verbatim_str.lower() not in ['no verbatim locality recorded', 'no locality recorded', 'nan', 'none', ''] and 
                    len(verbatim_str) > 5):
                    locality_to_try = verbatim_str
                    logger.info(f"Will try LLM with verbatim: '{verbatim_str}'")
            
            if locality_to_try:
                logger.info(f"Attempting LLM landmark extraction for: '{locality_to_try}'")
                landmark = self._extract_landmark_llm(locality_to_try, row['county'], row['stateProvince'], row['countryCode'])
                if landmark:
                    logger.info(f"Extracted landmark: {landmark}")
                    
                    # Try landmark with context first (more specific)
                    fallback_parts = [landmark]
                    for f in ['county', 'stateProvince', 'countryCode']:
                        if pd.notnull(row[f]) and str(row[f]).strip().lower() != 'unknown':
                            fallback_parts.append(str(row[f]))
                    fallback_string = ", ".join(fallback_parts)
                    logger.info(f"Trying LLM landmark with context: {fallback_string}")
                    try:
                        location = self.geocoder(fallback_string)
                        if location:
                            logger.info(f"LLM Success with context: {location.latitude}, {location.longitude}")
                            return location.latitude, location.longitude, True
                    except Exception as e:
                        logger.error(f"LLM context error: {e}")
                    
                    # Try landmark alone as fallback
                    logger.info(f"Trying LLM landmark alone: {landmark}")
                    try:
                        location = self.geocoder(landmark)
                        if location:
                            logger.info(f"LLM Success (landmark alone): {location.latitude}, {location.longitude}")
                            return location.latitude, location.longitude, True
                    except Exception as e:
                        logger.error(f"LLM landmark error: {e}")
                else:
                    logger.info("LLM landmark extraction returned None")
            else:
                logger.info("No valid locality/verbatim found for LLM extraction")
        else:
            logger.info("LLM is disabled, skipping landmark extraction")
        
        # Strategy 5: Municipality fallback (before last resort)
        if pd.notnull(row.get('municipality')):
            municipality_str = str(row['municipality']).strip()
            if (municipality_str.lower() not in ['nan', 'none', ''] and len(municipality_str) > 2):
                logger.info(f"Trying municipality fallback: '{municipality_str}'")
                # Try municipality alone first
                try:
                    location = self.geocoder(municipality_str)
                    if location:
                        logger.info(f"Municipality Success: {location.latitude}, {location.longitude}")
                        return location.latitude, location.longitude, False
                except Exception as e:
                    logger.error(f"Municipality error: {e}")
                
                # Try municipality with context
                municipality_parts = [municipality_str]
                for f in ['county', 'stateProvince', 'countryCode']:
                    if pd.notnull(row[f]):
                        municipality_parts.append(str(row[f]))
                municipality_with_context = ", ".join(municipality_parts)
                try:
                    location = self.geocoder(municipality_with_context)
                    if location:
                        logger.info(f"Municipality Success with context: {location.latitude}, {location.longitude}")
                        return location.latitude, location.longitude, False
                except Exception as e:
                    logger.error(f"Municipality context error: {e}")
                
                # Try LLM landmark extraction for municipality if it's long/complex
                if self.llm_enabled and len(municipality_str) > 20:  # Only for complex municipality strings
                    logger.info(f"Municipality is complex, trying LLM extraction: '{municipality_str}'")
                    landmark = self._extract_landmark_llm(municipality_str, row['county'], row['stateProvince'], row['countryCode'])
                    if landmark:
                        logger.info(f"Extracted municipality landmark: {landmark}")
                        
                        # Try landmark with context first (more specific)
                        municipality_llm_parts = [landmark]
                        for f in ['county', 'stateProvince', 'countryCode']:
                            if pd.notnull(row[f]) and str(row[f]).strip().lower() != 'unknown':
                                municipality_llm_parts.append(str(row[f]))
                        municipality_llm_with_context = ", ".join(municipality_llm_parts)
                        logger.info(f"Trying municipality LLM landmark with context: {municipality_llm_with_context}")
                        try:
                            location = self.geocoder(municipality_llm_with_context)
                            if location:
                                logger.info(f"Municipality LLM Success with context: {location.latitude}, {location.longitude}")
                                return location.latitude, location.longitude, True
                        except Exception as e:
                            logger.error(f"Municipality LLM context error: {e}")
                        
                        # Try landmark alone as fallback
                        logger.info(f"Trying municipality LLM landmark alone: {landmark}")
                        try:
                            location = self.geocoder(landmark)
                            if location:
                                logger.info(f"Municipality LLM Success (landmark alone): {location.latitude}, {location.longitude}")
                                return location.latitude, location.longitude, True
                        except Exception as e:
                            logger.error(f"Municipality LLM landmark error: {e}")
                    else:
                        logger.info("Municipality LLM landmark extraction returned None")
        
        # Strategy 6: Manual geocoding (before last resort)
        # Check if we have manual geocodes for this locality or municipality
        locality_key = row.get('locality') if pd.notnull(row.get('locality')) else None
        municipality_key = row.get('municipality') if pd.notnull(row.get('municipality')) else None
        
        for key in [locality_key, municipality_key]:
            if key and key in self.manual_geocodes:
                lat, lon = self.manual_geocodes[key]
                logger.info(f"Manual geocode found for '{key}': {lat}, {lon}")
                return lat, lon, False
        
        # Strategy 7: LAST RESORT - County-State only (very broad)
        if pd.notnull(row['county']) and pd.notnull(row['stateProvince']):
            county_state = f"{row['county']}, {row['stateProvince']}"
            if pd.notnull(row['countryCode']):
                county_state += f", {row['countryCode']}"
            logger.info(f"Trying LAST RESORT County-State: {county_state}")
            try:
                location = self.geocoder(county_state)
                if location:
                    logger.info(f"LAST RESORT Success: {location.latitude}, {location.longitude}")
                    return location.latitude, location.longitude, False
            except Exception as e:
                logger.error(f"Error in LAST RESORT attempt: {e}")
                results.append(("LAST RESORT County-State", str(e)))
        
        if results:
            logger.warning(f"All geocoding attempts failed for row {row.name}")
            for attempt_type, error in results:
                logger.warning(f"- {attempt_type}: {error}")
        
        return None, None, False

    def geocode_dataset(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        logger.info(f"Starting geocoding of {len(df)} records")
        result_df = df.copy()
        if 'decimalLatitude' not in result_df.columns:
            result_df['decimalLatitude'] = None
        if 'decimalLongitude' not in result_df.columns:
            result_df['decimalLongitude'] = None
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
                    result_df.at[idx, 'decimalLatitude'] = lat
                    result_df.at[idx, 'decimalLongitude'] = lon
                    result_df.at[idx, 'used_llm'] = used_llm
                time.sleep(0.1)
        successful_geocoding = result_df['decimalLatitude'].notna().sum()
        llm_used = result_df['used_llm'].sum()
        logger.info(f"Geocoding complete:")
        logger.info(f"  - Total records: {len(result_df)}")
        logger.info(f"  - Successfully geocoded: {successful_geocoding}")
        logger.info(f"  - Used LLM: {llm_used}")
        logger.info(f"  - Success rate: {successful_geocoding/len(result_df)*100:.1f}%")
        return result_df

    def get_geocoding_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_records = len(df)
        successful_geocoding = df['decimalLatitude'].notna().sum()
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
    for idx, row in df[df['decimalLatitude'].isnull() | df['decimalLongitude'].isnull()].iterrows():
        key = row.get('locality') or row.get('municipality')
        if key and key in manual_dict:
            df.at[idx, 'decimalLatitude'] = manual_dict[key]['lat']
            df.at[idx, 'decimalLongitude'] = manual_dict[key]['lon']
            logger.info(f"Applied manual geocode for {key} at index {idx}")
    return df 