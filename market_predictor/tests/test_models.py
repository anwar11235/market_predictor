"""
Test suite for model implementations and ensemble functionality
"""

import pytest
import pandas as pd
import numpy as np
from src.models import (
    ModelFactory,
    EnsembleModel,
    create_model,
    create_ensemble
)
from config import Config, load_validated_config

@pytest.fixture
def config():
    """Fixture for test configuration"""
    return load_validated_config('config/parameters.yaml')

@pytest.fixture
def sample_data():
    """Fixture for sample training data"""
    n_samples = 1000
    n_features = 20
    
    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate target (binary classification)
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y

@pytest.fixture
def model_factory(config):
    """Fixture for model factory"""
    return ModelFactory(config)

class TestModelFactory:
    """Test suite for model factory"""
    
    def test_model_creation(self, model_factory):
        """Test creation of different model types"""
        model_types = ['random_forest', 'xgboost', 'lightgbm']
        
        for model_type in model_types:
            model = model_factory.create_model(model_type)
            assert model is not None
            assert hasattr(model, 'train')
            assert hasattr(model, 'predict')
    
    def test_custom_params(self, model_factory):
        """Test model creation with custom parameters"""
        custom_params = {
            'n_estimators': 100,
            'max_depth': 5
        }
        
        model = model_factory.create_model('random_forest', custom_params)
        assert model.get_params()['n_estimators'] == 100
        assert model.get_params()['max_depth'] == 5
    
    def test_invalid_model_type(self, model_factory):
        """Test handling of invalid model type"""
        with pytest.raises(ValueError):
            model_factory.create_model('invalid_model_type')

class TestBaseModel:
    """Test suite for base model functionality"""
    
    def test_model_training(self, model_factory, sample_data):
        """Test model training functionality"""
        X, y = sample_data
        model = model_factory.create_model('random_forest')
        
        # Train model
        history = model.train(X, y)
        assert history is not None
        
        # Check if model is fitted
        assert hasattr(model.model, 'predict')
    
    def test_model_prediction(self, model_factory, sample_data):
        """Test model prediction functionality"""
        X, y = sample_data
        model = model_factory.create_model('random_forest')
        
        # Train and predict
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probability predictions
        prob_predictions = model.predict_proba(X)
        assert prob_predictions.shape == (len(y), 2)
        assert all((0 <= p <= 1).all() for p in prob_predictions)
    
    def test_model_validation(self, model_factory, sample_data):
        """Test model validation functionality"""
        X, y = sample_data
        model = model_factory.create_model('random_forest')
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Train with validation
        history = model.train(X_train, y_train, X_val, y_val)
        assert 'val_metrics' in history

class TestEnsembleModel:
    """Test suite for ensemble model functionality"""
    
    @pytest.fixture
    def base_models(self, model_factory, sample_data):
        """Fixture for trained base models"""
        X, y = sample_data
        models = {}
        
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            model = model_factory.create_model(model_type)
            model.train(X, y)
            models[model_type] = model
        
        return models
    
    def test_ensemble_creation(self, config, base_models):
        """Test ensemble model creation"""
        ensemble = create_ensemble(config, list(base_models.values()))
        assert isinstance(ensemble, EnsembleModel)
        assert len(ensemble.base_models) == len(base_models)
    
    def test_ensemble_prediction(self, config, base_models, sample_data):
        """Test ensemble prediction functionality"""
        X, y = sample_data
        ensemble = create_ensemble(config, list(base_models.values()))
        
        # Make predictions
        predictions = ensemble.predict(X)
        assert len(predictions) == len(y)
        
        # Test probability predictions
        prob_predictions = ensemble.predict_proba(X)
        assert prob_predictions.shape == (len(y), 2)
    
    def test_ensemble_weights(self, config, base_models, sample_data):
        """Test ensemble weight adjustment"""
        X, y = sample_data
        ensemble = create_ensemble(config, list(base_models.values()))
        
        # Set custom weights
        custom_weights = np.array([0.5, 0.3, 0.2])
        ensemble.set_weights(custom_weights)
        
        # Check if weights are properly set
        np.testing.assert_array_equal(ensemble.weights, custom_weights)

def test_model_save_load(model_factory, sample_data, tmp_path):
    """Test model saving and loading"""
    X, y = sample_data
    model = model_factory.create_model('random_forest')
    model.train(X, y)
    
    # Save model
    save_path = tmp_path / "model.joblib"
    model.save(save_path)
    
    # Load model
    loaded_model = create_model('random_forest')
    loaded_model.load(save_path)
    
    # Compare predictions
    np.testing.assert_array_equal(
        model.predict(X),
        loaded_model.predict(X)
    )

def test_model_feature_importance(model_factory, sample_data):
    """Test feature importance calculation"""
    X, y = sample_data
    model = model_factory.create_model('random_forest')
    model.train(X, y)
    
    importance = model.get_feature_importance()
    assert isinstance(importance, pd.Series)
    assert len(importance) == X.shape[1]
    assert all(importance >= 0)

if __name__ == '__main__':
    pytest.main([__file__])