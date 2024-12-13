"""
Market Predictor Configuration Module

This module provides configuration management and validation for the market prediction system.
"""

from .config import Config, DataConfig, TechnicalFeatureConfig, MacroFeatureConfig, SentimentFeatureConfig, ModelConfig
from .config_validator import ConfigValidator

__all__ = [
    'Config',
    'DataConfig',
    'TechnicalFeatureConfig',
    'MacroFeatureConfig',
    'SentimentFeatureConfig',
    'ModelConfig',
    'ConfigValidator'
]

# Version of the config module
__version__ = '0.1.0'

def load_validated_config(config_path: str) -> Config:
    """
    Load and validate configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Validated Config object
        
    Raises:
        ValueError: If configuration validation fails
    """
    validator = ConfigValidator()
    if not validator.validate_config_file(config_path):
        validator.print_validation_results()
        raise ValueError("Configuration validation failed")
    
    return Config(config_path)