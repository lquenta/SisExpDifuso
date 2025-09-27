"""
Configuration loader and saver for fuzzy systems.
Handles JSON serialization/deserialization with validation.
"""

import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from core.schema import FuzzySystemConfig, create_example_config


class FuzzyConfigLoader:
    """Loader and saver for fuzzy system configurations."""
    
    def __init__(self):
        """Initialize configuration loader."""
        pass
    
    def load_from_file(self, file_path: Union[str, Path]) -> FuzzySystemConfig:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Loaded fuzzy system configuration
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or configuration is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return self.load_from_dict(data)
    
    def load_from_dict(self, data: Dict[str, Any]) -> FuzzySystemConfig:
        """
        Load configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Loaded fuzzy system configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Validate and create configuration
            config = FuzzySystemConfig(**data)
            
            # Additional validation
            self._validate_configuration(config)
            
            return config
            
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")
    
    def load_from_json(self, json_str: str) -> FuzzySystemConfig:
        """
        Load configuration from JSON string.
        
        Args:
            json_str: JSON configuration string
            
        Returns:
            Loaded fuzzy system configuration
            
        Raises:
            ValueError: If JSON is invalid or configuration is invalid
        """
        try:
            data = json.loads(json_str)
            return self.load_from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    def save_to_file(self, config: FuzzySystemConfig, file_path: Union[str, Path], 
                    format: str = "json", indent: int = 2) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Fuzzy system configuration
            file_path: Path to save file
            format: File format ("json" or "yaml")
            indent: Indentation for JSON output
            
        Raises:
            ValueError: If format is unsupported
        """
        file_path = Path(file_path)
        
        # Convert to dictionary
        data = self.config_to_dict(config)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == "yaml":
                yaml.dump(data, f, default_flow_style=False, indent=indent)
            else:
                json.dump(data, f, indent=indent, ensure_ascii=False)
    
    def save_to_json(self, config: FuzzySystemConfig, indent: int = 2) -> str:
        """
        Save configuration to JSON string.
        
        Args:
            config: Fuzzy system configuration
            indent: JSON indentation
            
        Returns:
            JSON configuration string
        """
        data = self.config_to_dict(config)
        return json.dumps(data, indent=indent, ensure_ascii=False)
    
    def config_to_dict(self, config: FuzzySystemConfig) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Args:
            config: Fuzzy system configuration
            
        Returns:
            Configuration dictionary
        """
        return config.dict(by_alias=True)
    
    def _validate_configuration(self, config: FuzzySystemConfig) -> None:
        """
        Perform additional validation on configuration.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check universe coverage
        for var in config.variables:
            coverage = self._check_universe_coverage(var)
            if coverage["has_gaps"]:
                print(f"Warning: Variable {var.name} has gaps in universe coverage")
        
        # Check rule references
        errors = config.validate_rule_references()
        if errors:
            raise ValueError(f"Rule validation errors: {errors}")
    
    def _check_universe_coverage(self, var) -> Dict[str, Any]:
        """Check universe coverage for a variable."""
        from ..core.mfuncs import analyze_universe_coverage
        
        # Create temporary list with single variable
        temp_vars = [var]
        return analyze_universe_coverage(temp_vars, tuple(var.universe))
    
    def create_template(self, project_name: str = "New_Project") -> FuzzySystemConfig:
        """
        Create a template configuration.
        
        Args:
            project_name: Name for the new project
            
        Returns:
            Template configuration
        """
        return FuzzySystemConfig(
            schema_version="1.0",
            project=project_name,
            variables=[
                {
                    "name": "Input1",
                    "kind": "input",
                    "universe": [0.0, 10.0],
                    "terms": [
                        {"label": "Low", "mf": {"type": "tri", "a": 0.0, "b": 0.0, "c": 5.0}},
                        {"label": "High", "mf": {"type": "tri", "a": 5.0, "b": 10.0, "c": 10.0}}
                    ]
                },
                {
                    "name": "Output1",
                    "kind": "output",
                    "universe": [0.0, 100.0],
                    "defuzz": "centroid",
                    "terms": [
                        {"label": "Low", "mf": {"type": "tri", "a": 0.0, "b": 0.0, "c": 50.0}},
                        {"label": "High", "mf": {"type": "tri", "a": 50.0, "b": 100.0, "c": 100.0}}
                    ]
                }
            ],
            rules=[
                {
                    "id": "R1",
                    "if": "Input1 is Low",
                    "then": [{"variable": "Output1", "term": "Low"}],
                    "weight": 1.0,
                    "note": "Template rule"
                }
            ]
        )


def load_config(file_path: Union[str, Path]) -> FuzzySystemConfig:
    """Convenience function to load configuration from file."""
    loader = FuzzyConfigLoader()
    return loader.load_from_file(file_path)


def save_config(config: FuzzySystemConfig, file_path: Union[str, Path], 
               format: str = "json") -> None:
    """Convenience function to save configuration to file."""
    loader = FuzzyConfigLoader()
    loader.save_to_file(config, file_path, format)


def create_config_template(project_name: str = "New_Project") -> FuzzySystemConfig:
    """Convenience function to create template configuration."""
    loader = FuzzyConfigLoader()
    return loader.create_template(project_name)
