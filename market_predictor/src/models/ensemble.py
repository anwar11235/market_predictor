"""
Ensemble Model Module

This module implements ensemble methods for combining multiple models:
- Voting ensemble
- Weighted ensemble
- Dynamic weight adjustment
- Performance tracking
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

class EnsembleModel(BaseModel):
    """Ensemble model that combines multiple base models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ensemble model
        
        Args:
            config: Dictionary containing ensemble configuration
                - models: List of base models
                - weights: Optional list of model weights
                - voting: Type of voting ('hard' or 'soft')
                - dynamic_weights: Whether to use dynamic weight adjustment
        """
        super().__init__(config)
        
        self.base_models: List[BaseModel] = []
        self.weights = np.array(config.get('weights', None))
        self.voting = config.get('voting', 'soft')
        self.dynamic_weights = config.get('dynamic_weights', True)
        
        # Performance tracking
        self.model_performances: Dict[str, List[float]] = {}
        self.weight_history: List[np.ndarray] = []
    
    def add_model(self, model: BaseModel, weight: Optional[float] = None):
        """
        Add a base model to the ensemble
        
        Args:
            model: Model instance to add
            weight: Optional weight for the model
        """
        self.base_models.append(model)
        
        if weight is not None:
            if self.weights is None:
                self.weights = np.ones(len(self.base_models))
            self.weights[-1] = weight
            
        # Normalize weights
        if self.weights is not None:
            self.weights = self.weights / np.sum(self.weights)
    
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train all base models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            Dictionary containing training history
        """
        training_history = {}
        
        for i, model in enumerate(self.base_models):
            self.logger.info(f"Training model {i+1}/{len(self.base_models)}")
            
            # Train model
            history = model.train(X_train, y_train, X_val, y_val)
            
            # Store training history
            training_history[f'model_{i+1}'] = history
            
            # Update weights if using validation set
            if self.dynamic_weights and X_val is not None and y_val is not None:
                self._update_weights(X_val, y_val)
        
        self.metadata['training_history'] = training_history
        return training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions
        
        Args:
            X: Features to generate predictions for
            
        Returns:
            Array of predictions
        """
        if self.voting == 'hard':
            predictions = np.array([model.predict(X) for model in self.base_models])
            if self.weights is None:
                # Simple majority voting
                return np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int)).argmax(),
                    axis=0,
                    arr=predictions
                )
            else:
                # Weighted voting
                weighted_predictions = predictions * self.weights.reshape(-1, 1)
                return np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int), weights=self.weights).argmax(),
                    axis=0,
                    arr=weighted_predictions
                )
        else:  # soft voting
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions
        
        Args:
            X: Features to generate predictions for
            
        Returns:
            Array of probability predictions
        """
        probas = np.array([model.predict_proba(X) for model in self.base_models])
        
        if self.weights is None:
            # Simple averaging
            return np.mean(probas, axis=0)
        else:
            # Weighted averaging
            return np.average(probas, axis=0, weights=self.weights)
    
    def _update_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Update model weights based on validation performance
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        performances = []
        
        # Calculate performance for each model
        for model in self.base_models:
            metrics = model.evaluate(X_val, y_val)
            performances.append(metrics['accuracy'])  # Using accuracy as the metric
        
        # Update weights based on performance
        new_weights = np.array(performances)
        new_weights = new_weights / np.sum(new_weights)
        
        # Apply exponential moving average to smooth weight changes
        if self.weights is None:
            self.weights = new_weights
        else:
            alpha = 0.3  # smoothing factor
            self.weights = alpha * new_weights + (1 - alpha) * self.weights
        
        # Store weight history
        self.weight_history.append(self.weights.copy())
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get aggregated feature importance across all base models
        
        Returns:
            Series with feature importance scores
        """
        try:
            importance_matrices = []
            
            for model in self.base_models:
                importance = model.get_feature_importance()
                importance_matrices.append(importance)
            
            # Aggregate importance scores
            if self.weights is None:
                final_importance = np.mean(importance_matrices, axis=0)
            else:
                final_importance = np.average(
                    importance_matrices, 
                    axis=0, 
                    weights=self.weights
                )
            
            return pd.Series(
                final_importance,
                index=self.feature_names,
                name='feature_importance'
            )
            
        except NotImplementedError:
            self.logger.warning(
                "Feature importance not available for some base models"
            )
            return pd.Series()
    
    def get_model_weights(self) -> pd.Series:
        """
        Get current model weights
        
        Returns:
            Series with model weights
        """
        return pd.Series(
            self.weights,
            index=[f"Model_{i+1}" for i in range(len(self.base_models))],
            name='model_weights'
        )
    
    def get_weight_history(self) -> pd.DataFrame:
        """
        Get history of weight adjustments
        
        Returns:
            DataFrame with weight history
        """
        return pd.DataFrame(
            self.weight_history,
            columns=[f"Model_{i+1}" for i in range(len(self.base_models))]
        )
    
    def set_params(self, **params):
        """
        Set model parameters
        
        Args:
            **params: Parameters to set
        """
        valid_params = ['voting', 'weights', 'dynamic_weights']
        for param, value in params.items():
            if param in valid_params:
                setattr(self, param, value)
            else:
                self.logger.warning(f"Unknown parameter: {param}")

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    from .model_factory import ModelFactory
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Create model factory
    factory = ModelFactory(config)
    
    # Create base models
    rf_model = factory.create_model('random_forest')
    xgb_model = factory.create_model('xgboost')
    lgb_model = factory.create_model('lightgbm')
    
    # Create ensemble configuration
    ensemble_config = {
        'voting': 'soft',
        'dynamic_weights': True
    }
    
    # Create and configure ensemble
    ensemble = EnsembleModel(ensemble_config)
    ensemble.add_model(rf_model)
    ensemble.add_model(xgb_model)
    ensemble.add_model(lgb_model)
    
    # Train ensemble (assuming data is prepared)
    # ensemble.train(X_train, y_train, X_val, y_val)