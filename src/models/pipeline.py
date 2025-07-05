"""
Main machine learning pipeline for huckleberry habitat prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from typing import Tuple, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuckleberryPredictor:
    """
    Main predictor class for huckleberry habitat suitability.
    
    This class implements the best performing model pipeline found during
    the experimental phase, combining XGBoost with Bernoulli Naive Bayes
    in a stacking ensemble.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to a saved model file. If None, will train a new model.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_pipeline(self) -> Any:
        """
        Create the best performing pipeline from experiments.
        
        Returns:
            Fitted pipeline with optimal hyperparameters.
        """
        # Best pipeline from TPOT optimization
        pipeline = make_pipeline(
            StackingEstimator(
                estimator=XGBClassifier(
                    learning_rate=0.5,
                    max_depth=3,
                    min_child_weight=8,
                    n_estimators=100,
                    n_jobs=1,
                    subsample=1.0,
                    verbosity=0
                )
            ),
            BernoulliNB(alpha=10.0, fit_prior=False)
        )
        
        # Fix random state for reproducibility
        set_param_recursive(pipeline.steps, 'random_state', 42)
        
        return pipeline
    
    def prepare_data(self, data: pd.DataFrame, 
                    target_col: str = 'occurrence') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training/prediction.
        
        Args:
            data: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (features, target)
        """
        # Ensure target column exists
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Columns to exclude from training (matching improved notebook model)
        exclude_cols = [
            'gbifID', 'gridmet_lat', 'gridmet_lon', 'gridmet_date', 
            'decimalLatitude', 'decimalLongitude', 'datetime', 'parsed_datetime',
            'season', 'month', 'day'
        ]
        
        # Separate features and target
        features = data.drop([target_col] + 
                           [col for col in exclude_cols if col in data.columns], 
                           axis=1)
        target = data[target_col]
        
        # Store feature names for later use
        if self.feature_names is None:
            self.feature_names = features.columns.tolist()
        
        logger.info(f"Training features: {len(self.feature_names)} columns")
        logger.info(f"Excluded columns: "
                   f"{[col for col in exclude_cols if col in data.columns]}")
        
        return features, target
    
    def prepare_inference_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for inference, ensuring it has the same features as training.
        
        Args:
            data: Input DataFrame for inference
            
        Returns:
            DataFrame with features ready for prediction
        """
        if self.feature_names is None:
            raise ValueError("Model must be fitted before preparing inference data")
        
        # Columns to exclude from inference features (matching improved notebook model)
        exclude_cols = [
            'gbifID', 'gridmet_lat', 'gridmet_lon', 'gridmet_date', 
            'decimalLatitude', 'decimalLongitude', 'datetime', 'parsed_datetime',
            'season', 'month', 'day'
        ]
        
        # Remove excluded columns if they exist
        inference_data = data.drop([col for col in exclude_cols 
                                  if col in data.columns], axis=1)
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(inference_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features for inference: "
                           f"{missing_features}")
        
        # Select only the features used in training
        inference_data = inference_data[self.feature_names]
        
        logger.info(f"Inference data prepared: {inference_data.shape}")
        
        return inference_data
    
    def fit(self, data: pd.DataFrame, target_col: str = 'occurrence', 
            test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Fit the model to the training data.
        
        Args:
            data: Training data
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Preparing data for training...")
        features, target = self.prepare_data(data, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and fit pipeline
        logger.info("Training model...")
        self.model = self.create_pipeline()
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
        self.is_fitted = True
        logger.info(f"Model training completed. Test accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare the data for inference (remove excluded columns, select features)
        data_prepared = self.prepare_inference_data(data)
        data_scaled = self.scaler.transform(data_prepared)
        return self.model.predict(data_scaled)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for new data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare the data for inference (remove excluded columns, select features)
        data_prepared = self.prepare_inference_data(data)
        data_scaled = self.scaler.transform(data_prepared)
        
        # Get probabilities
        probabilities = self.model.predict_proba(data_scaled)
        
        return probabilities
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Try to get feature importance from XGBoost within the StackingEstimator
        try:
            # Access the XGBoost model within the StackingEstimator
            stacking_estimator = self.model.named_steps['stackingestimator']
            xgb_model = stacking_estimator.estimator
            importance = xgb_model.feature_importances_
            
            logger.info(f"Successfully extracted feature importance from XGBoost model")
            
        except (KeyError, AttributeError) as e:
            logger.warning(f"Could not extract feature importance from XGBoost: {e}")
            
            # Try alternative approaches
            try:
                # Try to access the model directly
                if hasattr(self.model, 'named_steps'):
                    for step_name, step in self.model.named_steps.items():
                        if hasattr(step, 'feature_importances_'):
                            importance = step.feature_importances_
                            logger.info(f"Found feature importance in step: {step_name}")
                            break
                    else:
                        raise AttributeError("No step with feature_importances_ found")
                else:
                    raise AttributeError("Model has no named_steps")
                    
            except (KeyError, AttributeError) as e2:
                logger.error(f"Could not extract feature importance from model: {e2}")
                # Return empty DataFrame with feature names for debugging
                return pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': [0.0] * len(self.feature_names)
                }).sort_values('importance', ascending=False)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Log some debugging info
        logger.info(f"Feature importance shape: {feature_importance.shape}")
        logger.info(f"Top 3 features: {feature_importance.head(3).to_dict('records')}")
        
        return feature_importance


class RandomForestPredictor:
    """
    Alternative Random Forest predictor for comparison.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize Random Forest predictor.
        
        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame, target_col: str = 'occurrence',
            test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Fit the Random Forest model.
        
        Args:
            data: Training data
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing training metrics
        """
        # Use the same prepare_data method as the main predictor
        features, target = self.prepare_data(data, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
        self.is_fitted = True
        
        return metrics
    
    def prepare_data(self, data: pd.DataFrame, 
                    target_col: str = 'occurrence') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training/prediction.
        
        Args:
            data: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (features, target)
        """
        # Ensure target column exists
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Columns to exclude from training (matching improved notebook model)
        exclude_cols = [
            'gbifID', 'gridmet_lat', 'gridmet_lon', 'gridmet_date', 
            'decimalLatitude', 'decimalLongitude', 'datetime', 'parsed_datetime',
            'season', 'month', 'day'
        ]
        
        # Separate features and target
        features = data.drop([target_col] + 
                           [col for col in exclude_cols if col in data.columns], 
                           axis=1)
        target = data[target_col]
        
        # Store feature names for later use
        if self.feature_names is None:
            self.feature_names = features.columns.tolist()
        
        logger.info(f"Training features: {len(self.feature_names)} columns")
        logger.info(f"Excluded columns: "
                   f"{[col for col in exclude_cols if col in data.columns]}")
        
        return features, target
    
    def prepare_inference_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for inference, ensuring it has the same features as training.
        
        Args:
            data: Input DataFrame for inference
            
        Returns:
            DataFrame with features ready for prediction
        """
        if self.feature_names is None:
            raise ValueError("Model must be fitted before preparing inference data")
        
        # Columns to exclude from inference features (matching improved notebook model)
        exclude_cols = [
            'gbifID', 'gridmet_lat', 'gridmet_lon', 'gridmet_date', 
            'decimalLatitude', 'decimalLongitude', 'datetime', 'parsed_datetime',
            'season', 'month', 'day'
        ]
        
        # Remove excluded columns if they exist
        inference_data = data.drop([col for col in exclude_cols 
                                  if col in data.columns], axis=1)
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(inference_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features for inference: "
                           f"{missing_features}")
        
        # Select only the features used in training
        inference_data = inference_data[self.feature_names]
        
        logger.info(f"Inference data prepared: {inference_data.shape}")
        
        return inference_data
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare the data for inference (remove excluded columns, select features)
        data_prepared = self.prepare_inference_data(data)
        data_scaled = self.scaler.transform(data_prepared)
        return self.model.predict(data_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance

    def save_model(self, filepath: str) -> None:
        """
        Save the trained Random Forest model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Load a trained Random Forest model from disk.
        
        Args:
            filepath: Path to the saved model file
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted'] 