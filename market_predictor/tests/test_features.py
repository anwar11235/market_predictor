"""
Test suite for feature engineering functionality
"""

import pytest
import pandas as pd
import numpy as np
from src.features.technical_features import TechnicalFeatures
from src.features.sentiment_features import SentimentFeatures
from src.features.macro_features import MacroFeatures
from src.features.feature_generator import FeatureGenerator
from config import Config, load_validated_config

@pytest.fixture
def config():
    """Fixture for test configuration"""
    return load_validated_config('config/parameters.yaml')

@pytest.fixture
def sample_market_data():
    """Fixture for sample market data"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    return pd.DataFrame({
        'Open': np.random.randn(len(dates)) * 10 + 100,
        'High': np.random.randn(len(dates)) * 10 + 101,
        'Low': np.random.randn(len(dates)) * 10 + 99,
        'Close': np.random.randn(len(dates)) * 10 + 100,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

@pytest.fixture
def feature_generators(config):
    """Fixture for feature generator instances"""
    return {
        'technical': TechnicalFeatures(config),
        'sentiment': SentimentFeatures(config),
        'macro': MacroFeatures(config),
        'generator': FeatureGenerator(config)
    }

class TestTechnicalFeatures:
    """Test suite for technical feature generation"""
    
    def test_technical_feature_generation(self, feature_generators, sample_market_data):
        """Test basic technical feature generation"""
        tech_features = feature_generators['technical'].calculate_all_features(sample_market_data)
        
        # Check basic structure
        assert isinstance(tech_features, pd.DataFrame)
        assert not tech_features.empty
        assert len(tech_features) == len(sample_market_data)
        
        # Check specific features exist
        expected_features = [
            'Returns', 'MA_5', 'RSI_14', 'ATR',
            'OBV', 'VWAP_Ratio'
        ]
        assert all(feature in tech_features.columns for feature in expected_features)
        
        # Check feature values
        assert tech_features['Returns'].notna().any()
        assert (tech_features['RSI_14'] >= 0).all() and (tech_features['RSI_14'] <= 100).all()
    
    def test_moving_averages(self, feature_generators, sample_market_data):
        """Test moving average calculations"""
        tech_features = feature_generators['technical'].calculate_all_features(sample_market_data)
        
        for window in [5, 20, 50]:
            ma_col = f'MA_{window}'
            assert ma_col in tech_features.columns
            assert tech_features[ma_col].notna().any()

class TestSentimentFeatures:
    """Test suite for sentiment feature generation"""
    
    @pytest.fixture
    def sample_sentiment_data(self):
        """Fixture for sample sentiment data"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        return pd.DataFrame({
            'text': ['positive news'] * len(dates),
            'source': ['news'] * len(dates)
        }, index=dates)
    
    def test_sentiment_feature_generation(self, feature_generators, sample_sentiment_data):
        """Test basic sentiment feature generation"""
        sent_features = feature_generators['sentiment'].calculate_all_features(sample_sentiment_data)
        
        assert isinstance(sent_features, pd.DataFrame)
        assert not sent_features.empty
        assert 'sentiment_score' in sent_features.columns
        assert (sent_features['sentiment_score'] >= -1).all() and (sent_features['sentiment_score'] <= 1).all()

class TestMacroFeatures:
    """Test suite for macroeconomic feature generation"""
    
    @pytest.fixture
    def sample_macro_data(self):
        """Fixture for sample macroeconomic data"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        return pd.DataFrame({
            'GDP': np.random.randn(len(dates)),
            'CPI': np.random.randn(len(dates)),
            'UNEMPLOYMENT': np.random.randn(len(dates))
        }, index=dates)
    
    def test_macro_feature_generation(self, feature_generators, sample_macro_data, sample_market_data):
        """Test basic macro feature generation"""
        macro_features = feature_generators['macro'].calculate_all_features(
            sample_macro_data, 
            sample_market_data
        )
        
        assert isinstance(macro_features, pd.DataFrame)
        assert not macro_features.empty
        assert len(macro_features.columns) > 0

class TestFeatureGenerator:
    """Test suite for main feature generator"""
    
    def test_feature_combination(self, feature_generators, sample_market_data):
        """Test combining features from different sources"""
        generator = feature_generators['generator']
        features = generator.generate_all_features(
            market_data=sample_market_data
        )
        
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert len(features.columns) > len(sample_market_data.columns)
    
    def test_feature_selection(self, feature_generators, sample_market_data):
        """Test feature selection functionality"""
        generator = feature_generators['generator']
        features = generator.generate_all_features(market_data=sample_market_data)
        
        # Create dummy target
        target = pd.Series(np.random.randint(0, 2, len(features)), index=features.index)
        
        selected_features = generator.select_features(features, target, n_features=10)
        
        assert isinstance(selected_features, pd.DataFrame)
        assert len(selected_features.columns) <= 10

def test_feature_consistency(feature_generators, sample_market_data):
    """Test consistency of feature generation"""
    # Generate features twice
    features1 = feature_generators['technical'].calculate_all_features(sample_market_data)
    features2 = feature_generators['technical'].calculate_all_features(sample_market_data)
    
    # Check if results are identical
    pd.testing.assert_frame_equal(features1, features2)

def test_feature_names(feature_generators, sample_market_data):
    """Test feature naming conventions"""
    features = feature_generators['technical'].calculate_all_features(sample_market_data)
    
    # Check if feature names are strings and unique
    assert all(isinstance(col, str) for col in features.columns)
    assert len(features.columns) == len(set(features.columns))

if __name__ == '__main__':
    pytest.main([__file__])