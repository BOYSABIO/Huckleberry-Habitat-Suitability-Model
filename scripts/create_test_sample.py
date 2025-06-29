#!/usr/bin/env python3
"""
Create a small test sample from occurrence data for pipeline testing.

This script creates a diverse sample of records to test different geocoding
scenarios:
- Records with existing coordinates
- Records that need traditional geocoding
- Records that might need LLM assistance
- Records that might need manual geocoding
- Records from different countries (US vs non-US)
"""

import pandas as pd
import random
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_occurrence_data(file_path: str) -> pd.DataFrame:
    """Load the occurrence data from the tab-separated file."""
    logger.info(f"Loading occurrence data from {file_path}")
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def create_diverse_sample(df: pd.DataFrame, sample_size: int = 30) -> pd.DataFrame:
    """
    Create a diverse sample for testing different geocoding scenarios.
    
    Args:
        df: Full occurrence dataframe
        sample_size: Number of records to sample
    
    Returns:
        Sample dataframe with diverse geocoding scenarios
    """
    logger.info(f"Creating diverse sample of {sample_size} records")
    
    # Define different categories for sampling
    categories = {
        'has_coordinates': [],
        'us_no_coordinates': [],
        'us_verbatim_locality': [],
        'us_county_state': [],
        'non_us': [],
        'needs_llm': []
    }
    
    # Categorize records
    for idx, row in df.iterrows():
        # Records with existing coordinates
        if (pd.notna(row.get('decimalLatitude')) and
                pd.notna(row.get('decimalLongitude'))):
            categories['has_coordinates'].append(idx)
        
        # US records without coordinates
        elif row.get('countryCode') == 'US':
            if (pd.notna(row.get('verbatimLocality')) and
                    len(str(row['verbatimLocality'])) > 10):
                categories['us_verbatim_locality'].append(idx)
            elif (pd.notna(row.get('county')) and
                  pd.notna(row.get('stateProvince'))):
                categories['us_county_state'].append(idx)
            else:
                categories['us_no_coordinates'].append(idx)
        
        # Non-US records
        elif row.get('countryCode') != 'US':
            categories['non_us'].append(idx)
    
    # Find records that might need LLM (complex locality descriptions)
    for idx, row in df.iterrows():
        if pd.notna(row.get('verbatimLocality')):
            verbatim = str(row['verbatimLocality']).lower()
            llm_indicators = ['mi.', 'miles', 'near', 'basin', 'range',
                             'mountain', 'lake', 'creek', 'river']
            if any(indicator in verbatim for indicator in llm_indicators):
                categories['needs_llm'].append(idx)
    
    # Calculate target samples per category
    total_categories = len([cat for cat in categories.values() if cat])
    samples_per_category = max(1, sample_size // total_categories)
    
    selected_indices = []
    
    # Sample from each category
    for category, indices in categories.items():
        if indices:
            # Take a random sample from this category
            category_sample = random.sample(
                indices, min(samples_per_category, len(indices))
            )
            selected_indices.extend(category_sample)
            logger.info(f"Selected {len(category_sample)} records from {category}")
    
    # If we don't have enough samples, add more from categories with more data
    if len(selected_indices) < sample_size:
        remaining_needed = sample_size - len(selected_indices)
        # Prioritize US records for testing
        us_indices = (categories['us_no_coordinates'] +
                     categories['us_verbatim_locality'] +
                     categories['us_county_state'])
        us_indices = [idx for idx in us_indices if idx not in selected_indices]
        
        if us_indices:
            additional_samples = random.sample(
                us_indices, min(remaining_needed, len(us_indices))
            )
            selected_indices.extend(additional_samples)
            logger.info(f"Added {len(additional_samples)} additional US records")
    
    # Create the sample dataframe
    sample_df = df.loc[selected_indices].copy()
    
    # Add a column to track the category for testing purposes
    sample_df['test_category'] = 'unknown'
    for category, indices in categories.items():
        sample_df.loc[sample_df.index.isin(indices), 'test_category'] = category
    
    logger.info(f"Created sample with {len(sample_df)} records")
    return sample_df


def save_test_sample(df: pd.DataFrame, output_path: str):
    """Save the test sample to a file."""
    logger.info(f"Saving test sample to {output_path}")
    df.to_csv(output_path, sep='\t', index=False)
    logger.info("Test sample saved successfully")


def print_sample_summary(df: pd.DataFrame):
    """Print a summary of the test sample."""
    logger.info("\n=== TEST SAMPLE SUMMARY ===")
    logger.info(f"Total records: {len(df)}")
    
    # Summary by category
    category_counts = df['test_category'].value_counts()
    for category, count in category_counts.items():
        logger.info(f"{category}: {count} records")
    
    # Summary by country
    country_counts = df['countryCode'].value_counts()
    logger.info("\nBy country:")
    for country, count in country_counts.items():
        logger.info(f"{country}: {count} records")
    
    # Summary of coordinate availability
    has_coords = df['decimalLatitude'].notna().sum()
    logger.info(f"\nRecords with coordinates: {has_coords}")
    logger.info(f"Records needing geocoding: {len(df) - has_coords}")
    
    # Show a few examples
    logger.info("\nSample records:")
    for idx, row in df.head(3).iterrows():
        locality = row.get('locality', 'N/A')
        verbatim = row.get('verbatimLocality', 'N/A')
        country = row.get('countryCode', 'N/A')
        category = row.get('test_category', 'N/A')
        
        # Handle NaN values and convert to string safely
        locality_str = str(locality) if pd.notna(locality) else 'N/A'
        verbatim_str = str(verbatim) if pd.notna(verbatim) else 'N/A'
        country_str = str(country) if pd.notna(country) else 'N/A'
        category_str = str(category) if pd.notna(category) else 'N/A'
        
        # Truncate verbatim if it's too long
        verbatim_display = verbatim_str[:50] + "..." if len(verbatim_str) > 50 else verbatim_str
        
        logger.info(f"  - {locality_str} | {verbatim_display} | {country_str} | {category_str}")


def main():
    """Main function to create the test sample."""
    # File paths
    input_file = "data/raw/occurrence.txt"
    output_file = "data/raw/occurrence_test_sample.txt"
    
    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found!")
        return
    
    try:
        # Load the data
        df = load_occurrence_data(input_file)
        
        # Create diverse sample
        sample_df = create_diverse_sample(df, sample_size=50)
        
        # Save the sample
        save_test_sample(sample_df, output_file)
        
        # Print summary
        print_sample_summary(sample_df)
        
        logger.info("\nTest sample created successfully!")
        logger.info(f"File saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating test sample: {e}")
        raise


if __name__ == "__main__":
    main() 