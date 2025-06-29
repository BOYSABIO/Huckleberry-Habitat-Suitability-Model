#!/usr/bin/env python3
"""
Test script to run the cleaning and processing pipeline with test data.

This script tests the data loading, preprocessing, and geocoding components
of the huckleberry habitat prediction pipeline using the test sample.
"""

import logging
import pandas as pd

# Import pipeline components
from src.data_load.loader import DataLoader
from src.data_preprocess.preprocessor import DataPreprocessor
from src.data_preprocess.geocode import Geocoder, load_manual_geocodes, apply_manual_geocodes
from src.data_validation.validate import validate_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_pipeline():
    """Test the cleaning and processing pipeline with test data."""
    
    logger.info("=== STARTING PIPELINE TEST WITH TEST SAMPLE ===")
    
    # 1. Load test data
    logger.info("\n1. Loading test data...")
    loader = DataLoader()
    try:
        df = loader.load_gbif_occurrences('data/raw/occurrence_test_sample.txt')
        logger.info(f"✅ Successfully loaded test data: {len(df)} records")
        logger.info(f"   Sample records by country: {df['countryCode'].value_counts().to_dict()}")
        logger.info(f"   Records with coordinates: {df['decimalLatitude'].notna().sum()}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load test data: {e}")
        return
    
    # 2. Preprocess data
    logger.info("\n2. Preprocessing data...")
    try:
        preprocessor = DataPreprocessor()
        df_cleaned = preprocessor.clean_occurrence_data(df)
        logger.info(f"✅ Successfully cleaned data: {len(df_cleaned)} records")
        
        # Show cleaning summary
        summary = preprocessor.get_cleaning_summary(df, df_cleaned)
        logger.info(f"   Records removed: {summary['records_removed']}")
        logger.info(f"   Retention rate: {summary['retention_rate']:.1f}%")
        
        if len(df_cleaned) == 0:
            logger.warning("⚠️  No records survived cleaning - this might indicate issues with the test data")
            return
            
    except Exception as e:
        logger.error(f"❌ Failed to preprocess data: {e}")
        return
    
    # 3. Temporal filtering (gridMET time bounds)
    logger.info("\n3. Temporal filtering...")
    try:
        df_temporal = preprocessor.filter_gridmet_time_bounds(df_cleaned)
        logger.info(f"✅ Successfully applied temporal filtering: {len(df_temporal)} records")
        if len(df_temporal) == 0:
            logger.warning("⚠️  No records survived temporal filtering - check date ranges!")
            return
    except Exception as e:
        logger.error(f"❌ Failed to apply temporal filtering: {e}")
        return
    
    # 4. Geocode data (if needed)
    logger.info("\n4. Geocoding data...")
    try:
        needs_geocoding = df_temporal['decimalLatitude'].isna().sum()
        logger.info(f"   Records needing geocoding: {needs_geocoding}")
        if needs_geocoding > 0:
            geocoder = Geocoder(llm_enabled=True)
            df_geocoded = geocoder.geocode_dataset(df_temporal)
            logger.info(f"✅ Successfully geocoded data: {len(df_geocoded)} records")
            geocoding_summary = geocoder.get_geocoding_summary(df_geocoded)
            logger.info(f"   Successfully geocoded: {geocoding_summary['successful_geocoding']}")
            logger.info(f"   Success rate: {geocoding_summary['success_rate']:.1f}%")
        else:
            logger.info("✅ All records already have coordinates - skipping geocoding")
            df_geocoded = df_temporal
    except Exception as e:
        logger.error(f"❌ Failed to geocode data: {e}")
        return
    
    # 5. Spatial filtering (gridMET spatial bounds)
    logger.info("\n5. Spatial filtering...")
    try:
        df_spatial = preprocessor.filter_gridmet_bounds(df_geocoded)
        logger.info(f"✅ Successfully applied spatial filtering: {len(df_spatial)} records")
        if len(df_spatial) == 0:
            logger.warning("⚠️  No records survived spatial filtering - check coordinates!")
            return
    except Exception as e:
        logger.error(f"❌ Failed to apply spatial filtering: {e}")
        return
    
    # 6. Apply manual geocodes (if available)
    logger.info("\n6. Applying manual geocodes...")
    try:
        manual_dict = load_manual_geocodes()
        df_final = apply_manual_geocodes(df_spatial, manual_dict)
        logger.info(f"✅ Successfully applied manual geocodes: {len(df_final)} records")
        # Add occurrence column (for real occurrences)
        df_final['occurrence'] = 1
        logger.info("Added 'occurrence' column (set to 1 for all records)")
    except Exception as e:
        logger.error(f"❌ Failed to apply manual geocodes: {e}")
        return

    # 7. Generate pseudo-absences and combine
    logger.info("\n7. Generating pseudo-absences and combining...")
    try:
        from src.data_preprocess.pseudoabsence import generate_pseudo_absences
        combined_df = generate_pseudo_absences(df_final, ratio=3, buffer_km=5, random_seed=42)
        logger.info(f"✅ Generated pseudo-absences: {sum(combined_df['occurrence'] == 0)} records")
        logger.info(f"   Total records after combining: {len(combined_df)}")
    except Exception as e:
        logger.error(f"❌ Failed to generate pseudo-absences: {e}")
        return

    # 8. Select essential columns for output
    logger.info("\n8. Selecting essential columns for output...")
    try:
        combined_df = preprocessor.select_columns(combined_df)
        logger.info(f"✅ Selected essential columns for output: {combined_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"❌ Failed to select essential columns: {e}")
        return

    # 9. Save processed dataset (before environmental data)
    logger.info("\n9. Saving processed dataset...")
    try:
        loader.save_processed_data(combined_df, 'huckleberry_processed.csv')
        logger.info("✅ Successfully saved processed dataset (before environmental data)")
    except Exception as e:
        logger.error(f"❌ Failed to save processed dataset: {e}")
        return

    # 10. Feature engineering (environmental data)
    logger.info("\n10. Feature engineering (environmental data)...")
    try:
        from src.features.environmental import EnvironmentalDataExtractor
        env = EnvironmentalDataExtractor()
        
        # Extract GridMET climate data
        combined_df = env.extract_gridmet_data(combined_df)
        logger.info(f"✅ GridMET data extracted: {len([col for col in combined_df.columns if col.startswith('air_') or col.startswith('precip') or col.startswith('specific_') or col.startswith('relative_') or col.startswith('mean_') or col.startswith('potential_') or col.startswith('surface_') or col.startswith('wind_')])} climate variables")
        
        # Add elevation data
        combined_df = env.add_elevation_data(combined_df)
        logger.info(f"✅ Elevation data added: {combined_df['elevation'].notna().sum()}/{len(combined_df)} records")
        
        # Add soil pH data
        combined_df = env.add_soil_data(combined_df)
        logger.info(f"✅ Soil pH data added: {combined_df['soil_ph'].notna().sum()}/{len(combined_df)} records")
        
    except Exception as e:
        logger.error(f"❌ Failed to add environmental data: {e}")
        return

    # 11. Validate final complete dataset
    logger.info("\n11. Validating final complete dataset...")
    try:
        validate_data(combined_df, expected_columns=['decimalLatitude', 'decimalLongitude', 'year', 'month', 'day', 'datetime', 'occurrence'])
        logger.info("✅ Data validation passed for complete enriched dataset")
    except Exception as e:
        logger.error(f"❌ Data validation failed: {e}")
        return
    
    # 12. Save final enriched dataset
    logger.info("\n12. Saving final enriched dataset...")
    try:
        loader.save_enriched_data(combined_df, 'huckleberry_final_enriched.csv')
        logger.info("✅ Successfully saved final enriched dataset")
        
    except Exception as e:
        logger.error(f"❌ Failed to save final enriched dataset: {e}")
        return
    
    # 13. Final summary
    logger.info("\n=== PIPELINE TEST SUMMARY ===")
    logger.info(f"✅ Pipeline completed successfully!")
    logger.info(f"   Original records: {len(df)}")
    logger.info(f"   Final records: {len(combined_df)}")
    logger.info(f"   Records with coordinates: {combined_df['decimalLatitude'].notna().sum()}")
    logger.info(f"   Pseudo-absences: {sum(combined_df['occurrence'] == 0)}")
    logger.info(f"   Environmental variables: {len([col for col in combined_df.columns if col not in ['decimalLatitude', 'decimalLongitude', 'datetime', 'occurrence', 'gbifID', 'year', 'month', 'day']])}")
    logger.info(f"   Ready for modeling: {'Yes' if len(combined_df) > 0 else 'No'}")
    
    # Show sample of final data
    logger.info(f"\nSample of final processed data:")
    sample_cols = ['decimalLatitude', 'decimalLongitude', 'occurrence', 'air_temperature', 'elevation', 'soil_ph']
    available_cols = [col for col in sample_cols if col in combined_df.columns]
    logger.info(combined_df[available_cols].head(3).to_string())


if __name__ == "__main__":
    test_pipeline() 