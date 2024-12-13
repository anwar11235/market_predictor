"""
Market Predictor Models Module

This module provides functionality for creating and managing different types of models:
- Base model interface
- Model factory for creating different model types
- Ensemble model implementation
"""

from .base_model import BaseModel
from .model_factory import ModelFactory
from .ensemble import EnsembleModel

__all__ = [
    'BaseModel',
    'ModelFactory',
    'EnsembleModel',
    'create_model',
    'create_ensemble'
]

# Version of the models module
__version__ = '0.1.0'

def create_model(config, model_type: str, custom_params: dict = None) -> BaseModel:
    """
    Convenience function to create a single model
    
    Args:
        config: Configuration object
        model_type: Type of model to create
        custom_params: Optional custom parameters for the model
        
    Returns:
        Initialized model instance
    """
    factory = ModelFactory(config)
    return factory.create_model(model_type, custom_params)

def create_ensemble(config, 
                   model_types: list,
                   ensemble_config: dict = None,
                   custom_params: dict = None) -> EnsembleModel:
    """
    Convenience function to create an ensemble of models
    
    Args:
        config: Configuration object
        model_types: List of model types to include in ensemble
        ensemble_config: Optional ensemble configuration
        custom_params: Optional dictionary of custom parameters for each model type
        
    Returns:
        Initialized ensemble model
    """
    # Default ensemble configuration
    default_ensemble_config = {
        'voting': 'soft',
        'dynamic_weights': True
    }
    
    # Update with provided config
    if ensemble_config:
        default_ensemble_config.update(ensemble_config)
    
    # Create factory and ensemble
    factory = ModelFactory(config)
    ensemble = EnsembleModel(default_ensemble_config)
    
    # Add each model to ensemble
    for model_type in model_types:
        model_params = custom_params.get(model_type) if custom_params else None
        model = factory.create_model(model_type, model_params)
        ensemble.add_model(model)
    
    return ensemble

# Common model configurations
MODEL_CONFIGS = {
    'classification': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1
        }
    },
    'ensemble': {
        'voting': {
            'voting': 'soft',
            'dynamic_weights': True
        },
        'stacking': {
            'voting': 'soft',
            'dynamic_weights': False
        }
    }
}