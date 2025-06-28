"""
Feature importance analysis for huckleberry habitat prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """
    Analyze and visualize feature importance for huckleberry prediction models.
    """
    
    def __init__(self):
        """Initialize the feature analyzer."""
        self.feature_importance_data = {}
    
    def add_model_importance(self, model_name: str, 
                           feature_importance: pd.DataFrame) -> None:
        """
        Add feature importance data from a model.
        
        Args:
            model_name: Name of the model
            feature_importance: DataFrame with 'feature' and 'importance' columns
        """
        self.feature_importance_data[model_name] = feature_importance
        logger.info(f"Added feature importance for {model_name}")
    
    def get_top_features(self, model_name: str, n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features for a model.
        
        Args:
            model_name: Name of the model
            n: Number of top features to return
            
        Returns:
            DataFrame with top N features
        """
        if model_name not in self.feature_importance_data:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.feature_importance_data[model_name].head(n)
    
    def compare_models(self, models: List[str], n_features: int = 10) -> pd.DataFrame:
        """
        Compare feature importance across multiple models.
        
        Args:
            models: List of model names to compare
            n_features: Number of top features to include
            
        Returns:
            DataFrame with comparison of feature importance
        """
        comparison_data = []
        
        for model_name in models:
            if model_name not in self.feature_importance_data:
                logger.warning(f"Model '{model_name}' not found, skipping")
                continue
            
            top_features = self.get_top_features(model_name, n_features)
            top_features['model'] = model_name
            comparison_data.append(top_features)
        
        if not comparison_data:
            raise ValueError("No valid models found for comparison")
        
        return pd.concat(comparison_data, ignore_index=True)
    
    def plot_feature_importance(self, model_name: str, n_features: int = 10,
                              figsize: tuple = (10, 6)) -> None:
        """
        Plot feature importance for a specific model.
        
        Args:
            model_name: Name of the model
            n_features: Number of top features to plot
            figsize: Figure size (width, height)
        """
        if model_name not in self.feature_importance_data:
            raise ValueError(f"Model '{model_name}' not found")
        
        top_features = self.get_top_features(model_name, n_features)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {n_features} Feature Importances - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, models: List[str], n_features: int = 10,
                            figsize: tuple = (12, 8)) -> None:
        """
        Plot feature importance comparison across models.
        
        Args:
            models: List of model names to compare
            n_features: Number of top features to include
            figsize: Figure size (width, height)
        """
        comparison_data = self.compare_models(models, n_features)
        
        plt.figure(figsize=figsize)
        
        # Create subplots for each model
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(models):
            if model_name not in self.feature_importance_data:
                continue
            
            model_data = comparison_data[comparison_data['model'] == model_name]
            top_features = model_data.head(n_features)
            
            axes[i].barh(range(len(top_features)), top_features['importance'])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'])
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()
    
    def get_common_features(self, models: List[str], 
                          n_features: int = 10) -> List[str]:
        """
        Find features that are important across multiple models.
        
        Args:
            models: List of model names to analyze
            n_features: Number of top features to consider per model
            
        Returns:
            List of common important features
        """
        feature_sets = []
        
        for model_name in models:
            if model_name in self.feature_importance_data:
                top_features = self.get_top_features(model_name, n_features)
                feature_sets.append(set(top_features['feature']))
        
        if not feature_sets:
            return []
        
        # Find intersection of all feature sets
        common_features = set.intersection(*feature_sets)
        return list(common_features)
    
    def export_importance_report(self, output_path: str, 
                               models: Optional[List[str]] = None) -> None:
        """
        Export feature importance report to CSV.
        
        Args:
            output_path: Path to save the report
            models: List of models to include. If None, includes all models.
        """
        if models is None:
            models = list(self.feature_importance_data.keys())
        
        report_data = []
        
        for model_name in models:
            if model_name in self.feature_importance_data:
                importance_data = self.feature_importance_data[model_name].copy()
                importance_data['model'] = model_name
                report_data.append(importance_data)
        
        if report_data:
            combined_report = pd.concat(report_data, ignore_index=True)
            combined_report.to_csv(output_path, index=False)
            logger.info(f"Feature importance report saved to {output_path}")
        else:
            logger.warning("No data to export")


def analyze_environmental_features(data: pd.DataFrame, 
                                 target_col: str = 'occurrence') -> pd.DataFrame:
    """
    Analyze which environmental features are most important.
    
    Args:
        data: Input DataFrame with environmental features
        target_col: Name of the target column
        
    Returns:
        DataFrame with environmental feature analysis
    """
    # Identify environmental features (you can customize this list)
    environmental_features = [
        'elevation', 'slope', 'aspect', 'temperature', 'precipitation',
        'humidity', 'solar_radiation', 'soil_type', 'land_cover'
    ]
    
    # Find features that exist in the data
    available_features = [f for f in environmental_features 
                         if f in data.columns]
    
    if not available_features:
        logger.warning("No environmental features found in data")
        return pd.DataFrame()
    
    # Calculate correlation with target
    correlations = []
    for feature in available_features:
        correlation = data[feature].corr(data[target_col])
        correlations.append({
            'feature': feature,
            'correlation': correlation,
            'abs_correlation': abs(correlation)
        })
    
    correlation_df = pd.DataFrame(correlations)
    correlation_df = correlation_df.sort_values('abs_correlation', 
                                               ascending=False)
    
    return correlation_df


def create_feature_summary(data: pd.DataFrame, 
                          target_col: str = 'occurrence') -> Dict:
    """
    Create a summary of features in the dataset.
    
    Args:
        data: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Dictionary with feature summary information
    """
    features = data.drop(target_col, axis=1)
    
    summary = {
        'total_features': len(features.columns),
        'feature_names': features.columns.tolist(),
        'data_types': features.dtypes.to_dict(),
        'missing_values': features.isnull().sum().to_dict(),
        'target_distribution': data[target_col].value_counts().to_dict()
    }
    
    return summary 