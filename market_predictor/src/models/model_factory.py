"""
Model Factory Module

This module handles the creation and initialization of different model types:
- Machine Learning Models (Random Forest, XGBoost, etc.)
- Deep Learning Models
- Custom Models
- Ensemble Models
"""

from typing import Dict, Any, Optional, Type
import logging
from pathlib import Path
import yaml
import importlib

from .base_model import BaseModel
from config import Config

class ModelFactory:
    """Factory class for creating model instances"""
    
    def __init__(self, config: Config):
        """
        Initialize ModelFactory with configuration
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.setup_logging()
        self._register_models()
    
    def setup_logging(self):
        """Configure logging for the model factory"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ModelFactory')
    
    def _register_models(self):
        """Register available model types"""
        self.registered_models = {
            # Traditional ML Models
            'random_forest': {
                'class': 'RandomForestModel',
                'module': 'sklearn.ensemble',
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'random_state': 42
                }
            },
            'xgboost': {
                'class': 'XGBClassifier',
                'module': 'xgboost',
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'lightgbm': {
                'class': 'LGBMClassifier',
                'module': 'lightgbm',
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': -1,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            
            # Deep Learning Models
            'lstm': {
                'class': 'LSTMModel',
                'module': 'src.models.deep_learning.lstm_model',
                'default_params': {
                    'units': 50,
                    'dropout': 0.2,
                    'batch_size': 32,
                    'epochs': 100
                }
            },
            
            # Ensemble Models
            'voting_ensemble': {
                'class': 'VotingEnsemble',
                'module': 'src.models.ensemble',
                'default_params': {
                    'voting': 'soft',
                    'weights': None
                }
            }
        }
    
    def create_model(self, 
                    model_type: str, 
                    custom_params: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Create and initialize a model instance
        
        Args:
            model_type: Type of model to create
            custom_params: Optional custom parameters for the model
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in self.registered_models:
            raise ValueError(
                f"Model type '{model_type}' not registered. "
                f"Available models: {list(self.registered_models.keys())}"
            )
        
        try:
            # Get model specifications
            model_specs = self.registered_models[model_type]
            
            # Import model class
            module = importlib.import_module(model_specs['module'])
            model_class = getattr(module, model_specs['class'])
            
            # Merge default and custom parameters
            params = model_specs['default_params'].copy()
            if custom_params:
                params.update(custom_params)
            
            # Initialize model
            model = model_class(params)
            
            self.logger.info(f"Created model of type '{model_type}'")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating model '{model_type}': {e}")
            raise
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a model type
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of default parameters
        """
        if model_type not in self.registered_models:
            raise ValueError(f"Model type '{model_type}' not registered")
        
        return self.registered_models[model_type]['default_params'].copy()
    
    def register_custom_model(self,
                            model_type: str,
                            model_class: Type[BaseModel],
                            default_params: Dict[str, Any]):
        """
        Register a custom model type
        
        Args:
            model_type: Name for the model type
            model_class: Model class to register
            default_params: Default parameters for the model
        """
        if model_type in self.registered_models:
            self.logger.warning(f"Overwriting existing model type '{model_type}'")
        
        self.registered_models[model_type] = {
            'class': model_class.__name__,
            'module': model_class.__module__,
            'default_params': default_params
        }
        
        self.logger.info(f"Registered custom model type '{model_type}'")
    
    def save_model_config(self, path: str):
        """
        Save model configurations to file
        
        Args:
            path: Path to save configurations
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare configuration for saving
        config = {
            model_type: {
                'module': specs['module'],
                'class': specs['class'],
                'default_params': specs['default_params']
            }
            for model_type, specs in self.registered_models.items()
        }
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"Saved model configurations to {path}")
    
    def load_model_config(self, path: str):
        """
        Load model configurations from file
        
        Args:
            path: Path to load configurations from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.registered_models = config
        self.logger.info(f"Loaded model configurations from {path}")

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Create model factory
    factory = ModelFactory(config)
    
    # Create a model with default parameters
    model = factory.create_model('random_forest')
    
    # Create a model with custom parameters
    custom_params = {
        'n_estimators': 200,
        'max_depth': 10
    }
    model_custom = factory.create_model('random_forest', custom_params)
    
    # Save model configurations
    factory.save_model_config('config/model_config.yaml')