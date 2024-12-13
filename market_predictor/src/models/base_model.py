"""
Base Model Module

This module defines the abstract base class that all models must implement.
It ensures consistent interface across different model types including:
- ML models (Random Forest, XGBoost, etc.)
- Deep Learning models
- Ensemble models
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime
import json
import pickle
from pathlib import Path

class BaseModel(ABC):
    """Abstract base class for all market prediction models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model with configuration
        
        Args:
            config: Dictionary containing model parameters
        """
        self.config = config
        self.model = None
        self.feature_names: List[str] = []
        self.setup_logging()
        
        # Model metadata
        self.metadata = {
            'model_type': self.__class__.__name__,
            'created_at': datetime.now().isoformat(),
            'training_history': [],
            'performance_metrics': {}
        }
    
    def setup_logging(self):
        """Configure logging for the model"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Dictionary containing training history
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Features to generate predictions for
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions
        
        Args:
            X: Features to generate predictions for
            
        Returns:
            Array of probability predictions
        """
        pass
    
    def evaluate(self, 
                X: pd.DataFrame, 
                y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Features to evaluate
            y: True targets
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            predictions = self.predict(X)
            probas = self.predict_proba(X)
            
            metrics = self._calculate_metrics(y, predictions, probas)
            self.metadata['performance_metrics'] = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def _calculate_metrics(self, 
                         y_true: pd.Series, 
                         y_pred: np.ndarray,
                         y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics
    
    def save(self, path: str):
        """
        Save model and metadata
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'metadata': self.metadata,
                    'feature_names': self.feature_names,
                    'config': self.config
                }, f)
            
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, path: str):
        """
        Load model and metadata
        
        Args:
            path: Path to load model from
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.metadata = data['metadata']
            self.feature_names = data['feature_names']
            self.config = data['config']
            
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance if model supports it
        
        Returns:
            Series with feature importance scores
        """
        raise NotImplementedError(
            "Feature importance not implemented for this model type"
        )
    
    def update_metadata(self, key: str, value: Any):
        """
        Update model metadata
        
        Args:
            key: Metadata key to update
            value: New value
        """
        self.metadata[key] = value
    
    @abstractmethod
    def set_params(self, **params):
        """
        Set model parameters
        
        Args:
            **params: Parameters to set
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        
        Returns:
            Dictionary of current parameters
        """
        return self.config
    
    def validate_features(self, X: pd.DataFrame):
        """
        Validate input features
        
        Args:
            X: Features to validate
        
        Raises:
            ValueError: If features are invalid
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Features must be a pandas DataFrame")
        
        if self.feature_names and set(X.columns) != set(self.feature_names):
            raise ValueError(
                f"Feature mismatch. Expected {self.feature_names}, "
                f"got {X.columns.tolist()}"
            )