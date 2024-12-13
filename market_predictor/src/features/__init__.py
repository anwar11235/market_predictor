"""
Market Predictor Features Module

This module provides functionality for generating and managing different types of features:
- Technical features
- Sentiment features
- Macro features
- Combined feature generation
"""

from .technical_features import TechnicalFeatures
from .sentiment_features import SentimentFeatures
from .macro_features import MacroFeatures
from .feature_generator import FeatureGenerator

__all__ = [
    'TechnicalFeatures',
    'SentimentFeatures',
    'MacroFeatures',
    'FeatureGenerator',
    'generate_all_features'
]

# Version of the features module
__version__ = '0.1.0'

def generate_all_features(config, market_data, macro_data=None, sentiment_data=None):
    """
    Convenience function to generate all features using the feature generator.
    
    Args:
        config: Configuration object containing feature parameters
        market_data: DataFrame with OHLCV data
        macro_data: Optional DataFrame with macroeconomic data
        sentiment_data: Optional DataFrame with sentiment data
        
    Returns:
        DataFrame containing all generated features
    """
    generator = FeatureGenerator(config)
    return generator.generate_all_features(
        market_data=market_data,
        macro_data=macro_data,
        sentiment_data=sentiment_data
    )

# Feature group names for easy access
FEATURE_GROUPS = {
    'technical': [
        'price_features',
        'volume_features',
        'momentum_features',
        'volatility_features',
        'pattern_features'
    ],
    'sentiment': [
        'market_sentiment',
        'news_sentiment',
        'social_sentiment',
        'technical_sentiment'
    ],
    'macro': [
        'economic_indicators',
        'interest_rates',
        'monetary_indicators',
        'market_environment'
    ]
}

# Default feature windows
DEFAULT_WINDOWS = {
    'ma_windows': [5, 20, 50],
    'volatility_windows': [5, 20],
    'momentum_windows': [7, 14, 28],
    'sentiment_windows': [1, 3, 7, 14]
}