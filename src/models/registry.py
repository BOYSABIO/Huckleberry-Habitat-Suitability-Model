"""
Model registry for versioning and managing trained models.
"""

import json
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.utils.logging_config import get_logger


class ModelRegistry:
    """Model registry for versioning and managing trained models."""
    
    def __init__(self, registry_path: str = "models/"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self.registry = self._load_registry()
        self.logger = get_logger("model_registry")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load existing registry data."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": [], "current": None}
    
    def _save_registry(self):
        """Save registry data to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        feature_names: List[str],
        training_data_info: Dict[str, Any],
        parameters: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a trained model.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            model_type: Type of model (e.g., 'random_forest', 'ensemble')
            metrics: Training metrics
            feature_names: List of feature names
            training_data_info: Information about training data
            parameters: Model parameters
            description: Model description
            tags: List of tags
            
        Returns:
            Model version ID
        """
        # Generate version ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"{model_name}_v{len(self.registry['models']) + 1}_{timestamp}"
        
        # Create model file path
        model_file = self.registry_path / f"{version_id}.joblib"
        
        # Save model
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'model_type': model_type,
            'parameters': parameters,
            'version_id': version_id
        }
        joblib.dump(model_data, model_file)
        
        # Create registry entry
        model_entry = {
            "version_id": version_id,
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "file_path": str(model_file),
            "metrics": metrics,
            "feature_names": feature_names,
            "training_data_info": training_data_info,
            "parameters": parameters,
            "description": description,
            "tags": tags or [],
            "status": "active"
        }
        
        # Add to registry
        self.registry["models"].append(model_entry)
        self.registry["current"] = version_id
        self._save_registry()
        
        self.logger.info(f"Registered model: {version_id}")
        return version_id
    
    def load_model(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model from the registry.
        
        Args:
            version_id: Model version ID (if None, loads current)
            
        Returns:
            Model data dictionary
        """
        if version_id is None:
            version_id = self.registry["current"]
        
        if version_id is None:
            raise ValueError("No current model available")
        
        # Find model entry
        model_entry = None
        for entry in self.registry["models"]:
            if entry["version_id"] == version_id:
                model_entry = entry
                break
        
        if model_entry is None:
            raise ValueError(f"Model version {version_id} not found")
        
        # Load model file
        model_file = Path(model_entry["file_path"])
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model_data = joblib.load(model_file)
        self.logger.info(f"Loaded model: {version_id}")
        return model_data
    
    def get_current_model(self) -> Optional[Dict[str, Any]]:
        """Get the current model entry."""
        if self.registry["current"] is None:
            return None
        
        for entry in self.registry["models"]:
            if entry["version_id"] == self.registry["current"]:
                return entry
        return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return self.registry["models"]
    
    def get_model_by_id(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific model by version ID."""
        for entry in self.registry["models"]:
            if entry["version_id"] == version_id:
                return entry
        return None
    
    def set_current_model(self, version_id: str):
        """Set the current model version."""
        # Verify model exists
        model_entry = self.get_model_by_id(version_id)
        if model_entry is None:
            raise ValueError(f"Model version {version_id} not found")
        
        self.registry["current"] = version_id
        self._save_registry()
        self.logger.info(f"Set current model to: {version_id}")
    
    def delete_model(self, version_id: str):
        """Delete a model from the registry."""
        # Find and remove from registry
        for i, entry in enumerate(self.registry["models"]):
            if entry["version_id"] == version_id:
                # Remove file
                model_file = Path(entry["file_path"])
                if model_file.exists():
                    model_file.unlink()
                
                # Remove from registry
                del self.registry["models"][i]
                
                # Update current if needed
                if self.registry["current"] == version_id:
                    if self.registry["models"]:
                        self.registry["current"] = self.registry["models"][-1]["version_id"]
                    else:
                        self.registry["current"] = None
                
                self._save_registry()
                self.logger.info(f"Deleted model: {version_id}")
                return
        
        raise ValueError(f"Model version {version_id} not found")
    
    def get_model_metrics(self, version_id: Optional[str] = None) -> Dict[str, float]:
        """Get metrics for a model version."""
        if version_id is None:
            version_id = self.registry["current"]
        
        if version_id is None:
            raise ValueError("No current model available")
        
        model_entry = self.get_model_by_id(version_id)
        if model_entry is None:
            raise ValueError(f"Model version {version_id} not found")
        
        return model_entry["metrics"]
    
    def compare_models(self, version_ids: List[str]) -> pd.DataFrame:
        """Compare multiple model versions."""
        if len(version_ids) < 2:
            raise ValueError("Need at least 2 model versions to compare")
        
        comparison_data = []
        for version_id in version_ids:
            model_entry = self.get_model_by_id(version_id)
            if model_entry is None:
                raise ValueError(f"Model version {version_id} not found")
            
            comparison_data.append({
                "version_id": version_id,
                "model_name": model_entry["model_name"],
                "model_type": model_entry["model_type"],
                "timestamp": model_entry["timestamp"],
                **model_entry["metrics"]
            })
        
        return pd.DataFrame(comparison_data)


# Convenience functions
def register_model(
    model,
    model_name: str,
    model_type: str,
    metrics: Dict[str, float],
    feature_names: List[str],
    training_data_info: Dict[str, Any],
    parameters: Dict[str, Any],
    description: str = "",
    tags: Optional[List[str]] = None
) -> str:
    """Convenience function to register a model."""
    registry = ModelRegistry()
    return registry.register_model(
        model=model,
        model_name=model_name,
        model_type=model_type,
        metrics=metrics,
        feature_names=feature_names,
        training_data_info=training_data_info,
        parameters=parameters,
        description=description,
        tags=tags
    )


def load_current_model() -> Dict[str, Any]:
    """Convenience function to load the current model."""
    registry = ModelRegistry()
    return registry.load_model() 