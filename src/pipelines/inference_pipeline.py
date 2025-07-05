"""
Inference pipeline for the Huckleberry Habitat Prediction Pipeline.
"""

import pandas as pd
import folium
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.config.settings import Settings
from src.utils.logging_config import get_logger, log_pipeline_step
from src.features.environmental import EnvironmentalDataExtractor
from src.models.registry import ModelRegistry
from src.data_validation.validate import validate_inference_data


class InferencePipeline:
    """Inference pipeline for making predictions."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize inference pipeline.
        
        Args:
            settings: Pipeline settings
        """
        self.settings = settings or Settings()
        self.logger = get_logger("inference_pipeline")
        self.model_registry = ModelRegistry(self.settings.model.model_registry_path)
        self.env_extractor = EnvironmentalDataExtractor()
        
        # Load current model
        self.model_data = None
        self.model = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load the current model from registry or specific file."""
        try:
            # Check if a specific model file is specified
            if self.settings.inference.model_file_path:
                import joblib
                model_file = Path(self.settings.inference.model_file_path)
                if not model_file.exists():
                    raise FileNotFoundError(f"Model file not found: {model_file}")
                
                loaded_data = joblib.load(model_file)
                
                # Handle different model file formats
                if isinstance(loaded_data, dict):
                    # Registry format: {'model': model, 'feature_names': [...], ...}
                    self.model_data = loaded_data
                    self.model = self.model_data['model']
                    self.feature_names = self.model_data['feature_names']
                    self.logger.info(f"Loaded registry-format model: {model_file}")
                elif isinstance(loaded_data, list) and len(loaded_data) > 0:
                    # Alternative format: [model, feature_names, ...]
                    self.model = loaded_data[0]
                    self.feature_names = loaded_data[1] if len(loaded_data) > 1 else []
                    self.model_data = {
                        'model': self.model,
                        'feature_names': self.feature_names,
                        'version_id': 'legacy_model'
                    }
                    self.logger.info(f"Loaded legacy-format model: {model_file}")
                    
                    # If feature_names is empty, try to get them from the model
                    if not self.feature_names and hasattr(self.model, 'feature_names_in_'):
                        self.feature_names = list(self.model.feature_names_in_)
                        self.model_data['feature_names'] = self.feature_names
                        self.logger.info(f"Extracted feature names from model: {self.feature_names}")
                else:
                    # Direct model format
                    self.model = loaded_data
                    self.feature_names = []
                    self.model_data = {
                        'model': self.model,
                        'feature_names': self.feature_names,
                        'version_id': 'direct_model'
                    }
                    self.logger.info(f"Loaded direct model: {model_file}")
                    
                    # Try to get feature names from the model
                    if hasattr(self.model, 'feature_names_in_'):
                        self.feature_names = list(self.model.feature_names_in_)
                        self.model_data['feature_names'] = self.feature_names
                        self.logger.info(f"Extracted feature names from model: {self.feature_names}")
            else:
                # Load from registry - handle production vs development
                if self.settings.model.model_name == "huckleberry_model_prod":
                    # For production, get the latest production model
                    self.model_data = self.model_registry.get_latest_model_by_name("huckleberry_model_prod")
                    if not self.model_data:
                        # Fallback to any available model if no production model exists
                        self.logger.warning("No production model found, falling back to latest available model")
                        self.model_data = self.model_registry.load_model()
                else:
                    # For development, get the latest model (or specific one if configured)
                    self.model_data = self.model_registry.load_model()
                
                self.model = self.model_data['model']
                self.feature_names = self.model_data['feature_names']
                self.logger.info(f"Loaded model: {self.model_data['version_id']}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _get_latest_gridmet_date(self) -> pd.Timestamp:
        """Get the latest available date from GridMET dataset."""
        try:
            # Load GridMET dataset to get time range
            ds = self.env_extractor.load_gridmet_dataset()
            latest_time = pd.to_datetime(ds.time.values.max())
            self.logger.info(f"Latest GridMET date: {latest_time}")
            return latest_time
        except Exception as e:
            self.logger.warning(f"Failed to get latest GridMET date: {e}")
            # Fallback to a reasonable date within GridMET range
            return pd.Timestamp('2020-12-31')
    
    @log_pipeline_step("Input Validation")
    def validate_input(
        self, 
        coordinates: List[Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Validate and prepare input data for inference.
        
        Args:
            coordinates: List of (lat, lon) tuples
            
        Returns:
            DataFrame with validated input data
        """
        self.logger.info(f"Validating input: {len(coordinates)} coordinates")
        
        # Create DataFrame from coordinates only
        # Let environmental extraction determine the best dates dynamically
        df = pd.DataFrame(coordinates, columns=['decimalLatitude', 'decimalLongitude'])
        
        self.logger.info("Coordinates prepared for environmental extraction")
        
        # Validate required columns
        is_valid = validate_inference_data(df, self.settings.inference.required_columns)
        if not is_valid:
            raise ValueError("Input validation failed")
        
        self.logger.info(f"Input validation passed: {len(df)} records")
        return df
    
    @log_pipeline_step("Environmental Data Extraction")
    def extract_environmental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract environmental data for inference coordinates."""
        self.logger.info("Extracting environmental data")
        
        # Determine target date for GridMET data
        target_date = None
        if not self.settings.inference.use_latest_gridmet and self.settings.inference.gridmet_date:
            target_date = self.settings.inference.gridmet_date
            self.logger.info(f"Using specified GridMET date: {target_date}")
        else:
            self.logger.info("Using latest available GridMET data")
        
        # Extract GridMET data
        df_with_gridmet = self.env_extractor.extract_gridmet_data(df, target_date=target_date)
        self.logger.info(f"Records with GridMET data: {len(df_with_gridmet)}/{len(df)}")
        
        if len(df_with_gridmet) == 0:
            raise ValueError("No coordinates within GridMET bounds")
        
        # Add elevation data
        df_with_elevation = self.env_extractor.add_elevation_data(df_with_gridmet)
        
        # Add soil data
        df_with_soil = self.env_extractor.add_soil_data(df_with_elevation)
        
        # Extract year, month, day from datetime (added by environmental extraction)
        if 'datetime' in df_with_soil.columns:
            df_with_soil['year'] = df_with_soil['datetime'].dt.year
            df_with_soil['month'] = df_with_soil['datetime'].dt.month
            df_with_soil['day'] = df_with_soil['datetime'].dt.day
            
            # Add season_num feature (same logic as training pipeline)
            def month_to_season_num(month):
                if month in [12, 1, 2]:
                    return 0  # Winter
                elif month in [3, 4, 5]:
                    return 1  # Spring
                elif month in [6, 7, 8]:
                    return 2  # Summer
                else:
                    return 3  # Fall
            df_with_soil['season_num'] = df_with_soil['month'].apply(month_to_season_num)
            self.logger.info(f"Date components and season extracted from GridMET datetime")
        
        self.logger.info(f"Environmental data extraction complete: {df_with_soil.shape}")
        return df_with_soil
    
    @log_pipeline_step("Feature Preparation")
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction."""
        self.logger.info("Preparing features for prediction")
        
        # Debug: Log available columns and expected features
        self.logger.info(f"Available columns: {list(df.columns)}")
        self.logger.info(f"Model expects features: {self.feature_names}")
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            self.logger.error(f"Missing features: {missing_features}")
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select only the features used by the model
        feature_df = df[self.feature_names].copy()
        
        # Handle any missing values
        feature_df = feature_df.fillna(0)  # Simple imputation for inference
        
        self.logger.info(f"Feature preparation complete: {feature_df.shape}")
        return feature_df
    
    @log_pipeline_step("Model Prediction")
    def make_predictions(self, feature_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the loaded model."""
        self.logger.info("Making predictions")
        
        # Make predictions
        predictions = self.model.predict(feature_df)
        probabilities = self.model.predict_proba(feature_df)
        
        # Create results DataFrame with original data and predictions
        results_df = original_df.copy()
        results_df['prediction'] = predictions
        results_df['probability'] = probabilities[:, 1]  # Probability of occurrence
        
        self.logger.info(f"Predictions complete: {len(results_df)} records")
        return results_df
    
    def create_prediction_map(
        self,
        results_df: pd.DataFrame,
        output_path: str = "outputs/maps/prediction_map.html",
        confidence_threshold: float = 0.8
    ) -> str:
        """
        Create an interactive map showing predictions.
        
        Args:
            results_df: DataFrame with predictions
            output_path: Path to save the map
            confidence_threshold: Minimum confidence for suitable habitat
            
        Returns:
            Path to the created map file
        """
        self.logger.info("Creating prediction map")
        
        # Filter for suitable habitat
        suitable_df = results_df[results_df['probability'] >= confidence_threshold].copy()
        
        if len(suitable_df) == 0:
            self.logger.warning("No suitable habitat found above confidence threshold")
            return None
        
        # Create map
        map_center = [
            suitable_df['decimalLatitude'].mean(),
            suitable_df['decimalLongitude'].mean()
        ]
        
        m = folium.Map(location=map_center, zoom_start=8)
        
        # Add markers for suitable habitat
        for _, row in suitable_df.iterrows():
            popup_text = f"""
            <b>Suitable Huckleberry Habitat</b><br>
            Confidence: {row['probability']:.2%}<br>
            Latitude: {row['decimalLatitude']:.4f}<br>
            Longitude: {row['decimalLongitude']:.4f}<br>
            """
            
            folium.Marker(
                location=[row['decimalLatitude'], row['decimalLongitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='green', icon='leaf')
            ).add_to(m)
        
        # Save map
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        
        self.logger.info(f"Prediction map saved to: {output_path}")
        return str(output_path)
    
    def save_predictions_csv(
        self,
        results_df: pd.DataFrame,
        output_path: str = "outputs/predictions/inference_predictions.csv"
    ) -> str:
        """
        Save predictions to CSV file.
        
        Args:
            results_df: DataFrame with predictions
            output_path: Path to save the CSV file
            
        Returns:
            Path to the saved CSV file
        """
        self.logger.info("Saving predictions to CSV")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Predictions saved to: {output_path}")
        return str(output_path)
    
    def generate_inference_summary(
        self, 
        results_df: pd.DataFrame, 
        confidence_threshold: float
    ) -> str:
        """Generate a comprehensive inference summary report."""
        self.logger.info("Generating inference summary report")
        
        try:
            # Calculate summary statistics
            total_coordinates = len(results_df)
            suitable_count = sum(results_df['probability'] >= confidence_threshold)
            avg_confidence = results_df['probability'].mean()
            
            # Confidence distribution
            high_confidence = sum(results_df['probability'] >= 0.8)
            medium_confidence = sum((results_df['probability'] >= 0.6) & (results_df['probability'] < 0.8))
            low_confidence = sum(results_df['probability'] < 0.6)
            
            # Top locations (highest confidence)
            top_locations = results_df.nlargest(5, 'probability')[
                ['decimalLatitude', 'decimalLongitude', 'probability']
            ].to_dict('records')
            
            # Create summary dictionary
            summary = {
                "inference_summary": {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "model_version": self.model_data['version_id'],
                    "total_coordinates": total_coordinates,
                    "valid_coordinates": total_coordinates,
                    "suitable_habitat_count": suitable_count,
                    "suitability_percentage": (suitable_count / total_coordinates) * 100,
                    "average_confidence": round(avg_confidence, 4),
                    "confidence_threshold": confidence_threshold,
                    "confidence_distribution": {
                        "high_confidence": high_confidence,
                        "medium_confidence": medium_confidence,
                        "low_confidence": low_confidence
                    },
                    "top_locations": top_locations
                }
            }
            
            # Save to JSON
            output_dir = Path("outputs/summaries")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            summary_path = output_dir / f"inference_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Inference summary saved: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return None
    
    def save_top_predictions(
        self, 
        results_df: pd.DataFrame, 
        confidence_threshold: float
    ) -> str:
        """Save high-confidence predictions to a separate CSV."""
        self.logger.info("Saving top predictions (high confidence)")
        
        try:
            # Filter for high-confidence predictions
            top_predictions = results_df[results_df['probability'] >= confidence_threshold].copy()
            
            if len(top_predictions) == 0:
                self.logger.warning("No high-confidence predictions found")
                return None
            
            # Sort by confidence (highest first)
            top_predictions = top_predictions.sort_values('probability', ascending=False)
            
            # Save to CSV
            output_dir = Path("outputs/predictions")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            top_predictions_path = output_dir / f"top_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            top_predictions.to_csv(top_predictions_path, index=False)
            
            self.logger.info(f"Top predictions saved: {top_predictions_path} ({len(top_predictions)} records)")
            return str(top_predictions_path)
            
        except Exception as e:
            self.logger.error(f"Top predictions save failed: {e}")
            return None
    
    def create_confidence_plot(self, results_df: pd.DataFrame) -> str:
        """Create a confidence distribution plot."""
        self.logger.info("Creating confidence distribution plot")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Confidence histogram
            ax1.hist(results_df['probability'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Confidence Distribution')
            ax1.grid(True, alpha=0.3)
            
            # 2. Confidence vs suitability
            colors = ['red' if p < 0.5 else 'green' for p in results_df['probability']]
            ax2.scatter(range(len(results_df)), results_df['probability'], c=colors, alpha=0.6)
            ax2.axhline(y=0.5, color='orange', linestyle='--', label='Suitability Threshold')
            ax2.set_xlabel('Location Index')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence by Location')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Confidence categories pie chart
            high_conf = sum(results_df['probability'] >= 0.8)
            med_conf = sum((results_df['probability'] >= 0.6) & (results_df['probability'] < 0.8))
            low_conf = sum(results_df['probability'] < 0.6)
            
            categories = ['High (≥0.8)', 'Medium (0.6-0.8)', 'Low (<0.6)']
            values = [high_conf, med_conf, low_conf]
            colors_pie = ['green', 'orange', 'red']
            
            ax3.pie(values, labels=categories, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Confidence Categories')
            
            # 4. Summary statistics
            stats_text = f"""
            Total Locations: {len(results_df)}
            Average Confidence: {results_df['probability'].mean():.3f}
            High Confidence (≥0.8): {high_conf}
            Medium Confidence (0.6-0.8): {med_conf}
            Low Confidence (<0.6): {low_conf}
            """
            
            ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title('Summary Statistics')
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            output_dir = Path("outputs/summaries")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plot_path = output_dir / f"confidence_plot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confidence plot saved: {plot_path}")
            return str(plot_path)
            
        except ImportError:
            self.logger.warning("matplotlib/seaborn not available, skipping plot")
            return None
        except Exception as e:
            self.logger.error(f"Plot creation failed: {e}")
            return None
    
    def run(
        self,
        coordinates: List[Tuple[float, float]],
        create_map: bool = True,
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Run the complete inference pipeline.
        
        Args:
            coordinates: List of (lat, lon) tuples
            create_map: Whether to create a prediction map
            confidence_threshold: Minimum confidence for suitable habitat
            
        Returns:
            Dictionary with inference results
        """
        self.logger.info("Starting inference pipeline")
        
        try:
            # Step 1: Validate input
            df = self.validate_input(coordinates)
            
            # Step 2: Extract environmental data
            df = self.extract_environmental_data(df)
            
            # Step 3: Prepare features
            feature_df = self.prepare_features(df)
            
            # Step 4: Make predictions
            results_df = self.make_predictions(feature_df, df)
            
            # Step 5: Save predictions to CSV
            csv_path = self.save_predictions_csv(results_df)
            
            # Step 6: Generate additional outputs
            summary_path = self.generate_inference_summary(results_df, confidence_threshold)
            top_predictions_path = self.save_top_predictions(results_df, confidence_threshold)
            confidence_plot_path = self.create_confidence_plot(results_df)
            
            # Step 7: Create map (optional)
            map_path = None
            if create_map:
                map_path = self.create_prediction_map(
                    results_df, confidence_threshold=confidence_threshold
                )
            
            # Step 7: Prepare results
            suitable_count = sum(results_df['probability'] >= confidence_threshold)
            avg_confidence = results_df['probability'].mean()
            
            self.logger.info("Inference pipeline completed successfully")
            
            return {
                "success": True,
                "total_coordinates": len(coordinates),
                "valid_coordinates": len(results_df),
                "suitable_habitat_count": suitable_count,
                "average_confidence": avg_confidence,
                "predictions": results_df,
                "csv_path": csv_path,
                "map_path": map_path,
                "summary_path": summary_path,
                "top_predictions_path": top_predictions_path,
                "confidence_plot_path": confidence_plot_path,
                "model_version": self.model_data['version_id']
            }
            
        except Exception as e:
            self.logger.error(f"Inference pipeline failed: {str(e)}")
            raise 