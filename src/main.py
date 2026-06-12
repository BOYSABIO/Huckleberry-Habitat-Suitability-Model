"""
Main entry point for the Huckleberry Habitat Prediction Pipeline.
"""

import argparse
import sys

from src.config.settings import Settings
from src.config.environments import DATASET_PRESETS, get_settings
from src.model.registry import MODEL_REGISTRY
from src.utils.logging_config import setup_logging
from src.inference import InferencePipeline
from src.training import TrainingPipeline

import src.model.implementations  # noqa: F401 — populate MODEL_REGISTRY


def setup_logging_for_run(settings: Settings):
    """Set up logging from settings."""
    return setup_logging(
        name="huckleberry_pipeline",
        level=settings.logging.level,
        log_file=settings.logging.log_file,
        format_string=settings.logging.format,
    )


def run_training_pipeline(
    sample: bool = False,
    dataset: str = None,
    model_type: str = None,
) -> dict:
    settings = get_settings(sample=sample, training_dataset=dataset, model_type=model_type)
    logger = setup_logging_for_run(settings)

    if dataset:
        logger.info("Training from dataset: %s", settings.data.training_dataset_path)
    else:
        logger.info("Training pipeline mode: %s", "sample" if sample else "full ETL")

    logger.info("Model type: %s", settings.model.model_type)

    training_pipeline = TrainingPipeline(settings)
    results = training_pipeline.run()

    logger.info("Training pipeline completed successfully")
    return results


def run_inference_pipeline(
    coordinates: list,
    create_map: bool = True,
    confidence_threshold: float = 0.8,
    gridmet_date: str = None,
    model_path: str = None,
) -> dict:
    settings = get_settings()
    if model_path:
        settings.inference.model_file_path = model_path

    if gridmet_date:
        settings.inference.use_latest_gridmet = False
        settings.inference.gridmet_date = gridmet_date
    else:
        settings.inference.use_latest_gridmet = True
        settings.inference.gridmet_date = None

    logger = setup_logging_for_run(settings)
    logger.info("Starting inference pipeline")

    inference_pipeline = InferencePipeline(settings)
    results = inference_pipeline.run(
        coordinates=coordinates,
        create_map=create_map,
        confidence_threshold=confidence_threshold,
    )

    logger.info("Inference pipeline completed successfully")
    return results


def main():
    """Main entry point."""
    model_types = sorted(MODEL_REGISTRY.keys())
    dataset_help = (
        "Pre-enriched CSV path or preset name (skips ETL). "
        f"Presets: {', '.join(sorted(DATASET_PRESETS.keys()))}. "
        "Example: --dataset hb"
    )

    parser = argparse.ArgumentParser(
        description="Huckleberry Habitat Prediction Pipeline"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    train_parser = subparsers.add_parser('train', help='Run training pipeline')
    train_parser.add_argument(
        '--sample',
        action='store_true',
        help='Run full ETL on the small GBIF sample (~15 records)',
    )
    train_parser.add_argument(
        '--dataset',
        type=str,
        metavar='PATH_OR_PRESET',
        help=dataset_help,
    )
    train_parser.add_argument(
        '--model-type',
        type=str,
        choices=model_types,
        default='random_forest',
        help=f"Registered model type to train (default: random_forest). Options: {', '.join(model_types)}",
    )

    infer_parser = subparsers.add_parser('infer', help='Run inference pipeline')
    infer_parser.add_argument(
        '--coordinates',
        nargs='+',
        type=float,
        required=True,
        help='Coordinates as lat1 lon1 lat2 lon2 ...',
    )
    infer_parser.add_argument(
        '--gridmet-date',
        type=str,
        help='Specific date for GridMET data (YYYY-MM-DD format, e.g., 2020-07-15)',
    )
    infer_parser.add_argument(
        '--model',
        type=str,
        help='Path to a trained .joblib file (default: registry current / v13)',
    )
    infer_parser.add_argument(
        '--no-map',
        action='store_true',
        help='Skip creating prediction map',
    )
    infer_parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.8,
        help='Minimum confidence for suitable habitat',
    )

    args = parser.parse_args()

    if args.command == 'train':
        try:
            results = run_training_pipeline(
                sample=args.sample,
                dataset=args.dataset,
                model_type=args.model_type,
            )
            print("Training completed successfully!")
            print(f"Model version: {results['model_version_id']}")
            print(f"Model type: {args.model_type}")
            print(f"Final record count: {results['final_record_count']}")
            print(f"Metrics: {results['metrics']}")

        except Exception as e:
            print(f"Training failed: {str(e)}")
            sys.exit(1)

    elif args.command == 'infer':
        try:
            if len(args.coordinates) % 2 != 0:
                raise ValueError("Coordinates must be pairs of lat, lon")

            coord_tuples = [
                (args.coordinates[i], args.coordinates[i + 1])
                for i in range(0, len(args.coordinates), 2)
            ]

            results = run_inference_pipeline(
                coordinates=coord_tuples,
                create_map=not args.no_map,
                confidence_threshold=args.confidence_threshold,
                gridmet_date=args.gridmet_date,
                model_path=args.model,
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
            else:
                print("Map was not created (no valid coordinates to plot)")

        except Exception as e:
            print(f"Inference failed: {str(e)}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
