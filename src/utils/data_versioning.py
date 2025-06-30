"""
Data versioning and lineage tracking utilities.
"""

import hashlib
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class DataVersioning:
    """Data versioning and lineage tracking."""
    
    def __init__(self, version_file: str = "data/versions.json"):
        """
        Initialize data versioning.
        
        Args:
            version_file: Path to version tracking file
        """
        self.version_file = Path(version_file)
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load existing version data."""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {"versions": [], "current": None}
    
    def _save_versions(self):
        """Save version data to file."""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def create_version(
        self,
        data_hash: str,
        description: str,
        input_files: List[str],
        output_files: List[str],
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new data version.
        
        Args:
            data_hash: Hash of the data
            description: Description of the transformation
            input_files: List of input file paths
            output_files: List of output file paths
            parameters: Parameters used in the transformation
            metadata: Additional metadata
            
        Returns:
            Version ID
        """
        version_id = f"v{len(self.versions['versions']) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version_data = {
            "id": version_id,
            "timestamp": datetime.now().isoformat(),
            "data_hash": data_hash,
            "description": description,
            "input_files": input_files,
            "output_files": output_files,
            "parameters": parameters,
            "metadata": metadata or {}
        }
        
        self.versions["versions"].append(version_data)
        self.versions["current"] = version_id
        self._save_versions()
        
        return version_id
    
    def get_data_hash(self, df: pd.DataFrame) -> str:
        """
        Generate a hash for the DataFrame.
        
        Args:
            df: DataFrame to hash
            
        Returns:
            Hash string
        """
        # Convert DataFrame to string representation and hash it
        data_str = df.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def track_transformation(
        self,
        df: pd.DataFrame,
        description: str,
        input_files: List[str],
        output_files: List[str],
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track a data transformation.
        
        Args:
            df: Transformed DataFrame
            description: Description of the transformation
            input_files: List of input file paths
            output_files: List of output file paths
            parameters: Parameters used in the transformation
            metadata: Additional metadata
            
        Returns:
            Version ID
        """
        data_hash = self.get_data_hash(df)
        return self.create_version(
            data_hash=data_hash,
            description=description,
            input_files=input_files,
            output_files=output_files,
            parameters=parameters,
            metadata=metadata
        )
    
    def get_current_version(self) -> Optional[Dict[str, Any]]:
        """Get the current version data."""
        if self.versions["current"] is None:
            return None
        
        for version in self.versions["versions"]:
            if version["id"] == self.versions["current"]:
                return version
        return None
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get the complete version history."""
        return self.versions["versions"]
    
    def get_version_by_id(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific version by ID."""
        for version in self.versions["versions"]:
            if version["id"] == version_id:
                return version
        return None


def create_data_version(
    df: pd.DataFrame,
    description: str,
    input_files: List[str],
    output_files: List[str],
    parameters: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to create a data version.
    
    Args:
        df: Transformed DataFrame
        description: Description of the transformation
        input_files: List of input file paths
        output_files: List of output file paths
        parameters: Parameters used in the transformation
        metadata: Additional metadata
        
    Returns:
        Version ID
    """
    versioning = DataVersioning()
    return versioning.track_transformation(
        df=df,
        description=description,
        input_files=input_files,
        output_files=output_files,
        parameters=parameters,
        metadata=metadata
    ) 