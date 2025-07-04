"""
Main entry point for the Huckleberry Habitat Prediction Pipeline.
"""

import argparse
import sys

from src.config.settings import Settings
from src.config.environments import get_settings
from src.utils.logging_config import setup_logging
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline


def setup_environment(environment: str = None) -> tuple[Settings, object]:
    """
    Set up the environment and logging.
    
    Args:
        environment: Environment name
        
    Returns:
        Tuple of (settings, logger)
    """
    # Get settings for environment
    settings = get_settings(environment)
    
    # Set up logging
    logger = setup_logging(
        name="huckleberry_pipeline",
        level=settings.logging.level,
        log_file=settings.logging.log_file,
        format_string=settings.logging.format
    )
    
    return settings, logger


def run_training_pipeline(environment: str = None) -> dict:
    """
    Run the training pipeline.
    
    Args:
        environment: Environment name
        
    Returns:
        Training results
    """
    settings, logger = setup_environment(environment)
    
    logger.info("Starting training pipeline")
    
    # Initialize and run training pipeline
    training_pipeline = TrainingPipeline(settings)
    results = training_pipeline.run()
    
    logger.info("Training pipeline completed successfully")
    return results


def run_inference_pipeline(
    coordinates: list,
    environment: str = None,
    create_map: bool = True,
    confidence_threshold: float = 0.8,
    gridmet_date: str = None
) -> dict:
    """
    Run the inference pipeline.
    
    Args:
        coordinates: List of (lat, lon) tuples
        environment: Environment name
        create_map: Whether to create a prediction map
        confidence_threshold: Minimum confidence for suitable habitat
        gridmet_date: Specific date for GridMET data (YYYY-MM-DD format)
        
    Returns:
        Inference results
    """
    settings, logger = setup_environment(environment)
    
    # Configure GridMET date settings
    if gridmet_date:
        settings.inference.use_latest_gridmet = False
        settings.inference.gridmet_date = gridmet_date
        logger.info(f"Using specified GridMET date: {gridmet_date}")
    else:
        settings.inference.use_latest_gridmet = True
        settings.inference.gridmet_date = None
        logger.info("Using latest available GridMET data")
    
    logger.info("Starting inference pipeline")

    # Initialize and run inference pipeline
    inference_pipeline = InferencePipeline(settings)
    results = inference_pipeline.run(
        coordinates=coordinates,
        create_map=create_map,
        confidence_threshold=confidence_threshold
    )
    
    logger.info("Inference pipeline completed successfully")
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Huckleberry Habitat Prediction Pipeline"
    )

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run training pipeline')
    train_parser.add_argument(
        '--environment',
        choices=['development', 'production', 'testing', 'test_sample'],
        default='development',
        help='Environment to run in'
    )
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference pipeline')
    infer_parser.add_argument(
        '--coordinates',
        nargs='+',
        type=float,
        required=True,
        help='Coordinates as lat1 lon1 lat2 lon2 ...'
    )
    infer_parser.add_argument(
        '--gridmet-date',
        type=str,
        help='Specific date for GridMET data (YYYY-MM-DD format, e.g., 2020-07-15)'
    )
    infer_parser.add_argument(
        '--environment',
        choices=['development', 'production', 'testing', 'test_sample'],
        default='development',
        help='Environment to run in'
    )
    infer_parser.add_argument(
        '--no-map',
        action='store_true',
        help='Skip creating prediction map'
    )
    infer_parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.8,
        help='Minimum confidence for suitable habitat'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'train':
        try:
            results = run_training_pipeline(args.environment)
            print("Training completed successfully!")
            print(f"Model version: {results['model_version_id']}")
            print(f"Final record count: {results['final_record_count']}")
            print(f"Metrics: {results['metrics']}")
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            sys.exit(1)
    
    elif args.command == 'infer':
        try:
            # Convert coordinates to tuples
            if len(args.coordinates) % 2 != 0:
                raise ValueError("Coordinates must be pairs of lat, lon")
            
            coord_tuples = [
                (args.coordinates[i], args.coordinates[i + 1])
                for i in range(0, len(args.coordinates), 2)
            ]
            
            results = run_inference_pipeline(
                coordinates=coord_tuples,
                environment=args.environment,
                create_map=not args.no_map,
                confidence_threshold=args.confidence_threshold,
                gridmet_date=args.gridmet_date
            )
            
            print("Inference completed successfully!")
            if args.gridmet_date:
                print(f"GridMET date used: {args.gridmet_date}")
            else:
                print("GridMET date used: Latest available")
            print(f"Total coordinates: {results['total_coordinates']}")
            print(f"Valid coordinates: {results['valid_coordinates']}")
            print(f"Suitable habitat count: {results['suitable_habitat_count']}")
            print(f"Average confidence: {results['average_confidence']:.2%}")
            print(f"Predictions saved to: {results['csv_path']}")
            
            if results['map_path']:
                print(f"Map saved to: {results['map_path']}")
            
        except Exception as e:
            print(f"Inference failed: {str(e)}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 