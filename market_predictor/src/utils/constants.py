"""
Constants Module

This module contains all constant values used throughout the project:
- Data configurations
- Model parameters
- Feature definitions
- Trading parameters
"""

from enum import Enum
from typing import Dict, List

# Time-related constants
class TimeFrame(Enum):
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    YEARLY = 'yearly'

# Market hours (EST)
MARKET_HOURS = {
    'open': '09:30',
    'close': '16:00',
    'pre_market_start': '04:00',
    'post_market_end': '20:00'
}

# Feature Groups
TECHNICAL_FEATURES = {
    'price': [
        'Returns',
        'Log_Returns',
        'Daily_Volatility'
    ],
    'moving_averages': [
        'MA_5', 'MA_20', 'MA_50',
        'MA_5_Slope', 'MA_20_Slope', 'MA_50_Slope',
        'Price_Distance_5', 'Price_Distance_20', 'Price_Distance_50'
    ],
    'volume': [
        'OBV',
        'Volume_Ratio',
        'Force_Index_EMA13'
    ],
    'volatility': [
        'High_Low_Range',
        'Daily_Gap',
        'ATR'
    ],
    'momentum': [
        'RSI_7', 'RSI_14', 'RSI_28',
        'MFI'
    ],
    'patterns': [
        'Higher_High',
        'Lower_Low',
        'Inside_Day'
    ],
    'trend': [
        'VWAP_Ratio',
        'New_Highs',
        'New_Lows',
        'Bullish_Momentum',
        'Bearish_Momentum',
        'Trend_Strength'
    ]
}

# Macro Economic Indicators
MACRO_INDICATORS = {
    'growth': [
        'GDP',
        'Industrial_Production',
        'Retail_Sales'
    ],
    'employment': [
        'Unemployment_Rate',
        'Non_Farm_Payrolls',
        'Initial_Jobless_Claims'
    ],
    'inflation': [
        'CPI',
        'PPI',
        'PCE'
    ],
    'monetary': [
        'Fed_Funds_Rate',
        'M2_Money_Supply',
        'Fed_Balance_Sheet'
    ],
    'market': [
        'Treasury_10Y',
        'Treasury_2Y',
        'VIX'
    ]
}

# Sentiment Sources
SENTIMENT_SOURCES = {
    'news': [
        'newsapi',
        'alpha_vantage',
        'finnhub'
    ],
    'social': [
        'reddit',
        'twitter'
    ]
}

# Model Parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8
    },
    'lstm': {
        'units': 50,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'optimizer': 'adam'
    }
}

# Evaluation Metrics
METRICS = {
    'classification': [
        'accuracy',
        'precision',
        'recall',
        'f1_score'
    ],
    'regression': [
        'mse',
        'rmse',
        'mae',
        'r2_score'
    ]
}

# Data Split Ratios
SPLIT_RATIOS = {
    'train': 0.7,
    'validation': 0.15,
    'test': 0.15
}

# Cache Settings
CACHE_CONFIG = {
    'enabled': True,
    'directory': 'data/cache',
    'expiry': {
        'market_data': 24,  # hours
        'macro_data': 168,  # hours (1 week)
        'sentiment_data': 4  # hours
    }
}

# API Rate Limits
RATE_LIMITS = {
    'newsapi': {
        'requests_per_day': 100,
        'requests_per_second': 1
    },
    'alpha_vantage': {
        'requests_per_minute': 5,
        'requests_per_day': 500
    },
    'finnhub': {
        'requests_per_minute': 60,
        'requests_per_second': 1
    },
    'reddit': {
        'requests_per_minute': 30,
        'requests_per_second': 1
    },
    'twitter': {
        'requests_per_hour': 500,
        'requests_per_second': 1
    }
}

# Error Messages
ERROR_MESSAGES = {
    'api': {
        'rate_limit': 'API rate limit exceeded. Please wait before making more requests.',
        'authentication': 'Authentication failed. Please check your API credentials.',
        'connection': 'Failed to connect to the API. Please check your internet connection.'
    },
    'data': {
        'missing': 'Required data is missing.',
        'invalid_format': 'Data format is invalid.',
        'date_range': 'Invalid date range specified.'
    },
    'model': {
        'training': 'Error occurred during model training.',
        'prediction': 'Error occurred during prediction.',
        'validation': 'Model validation failed.'
    }
}

# Feature Engineering Parameters
FEATURE_PARAMS = {
    'moving_average_windows': [5, 20, 50],
    'volatility_windows': [5, 20],
    'momentum_windows': [7, 14, 28],
    'volume_windows': [5, 10, 20],
    'sentiment_windows': [1, 3, 7, 14]
}

# Trading Parameters
TRADING_PARAMS = {
    'position_sizes': {
        'max_position': 0.1,  # 10% of portfolio
        'min_position': 0.01  # 1% of portfolio
    },
    'stop_loss': {
        'default': 0.02,  # 2%
        'trailing': 0.03  # 3%
    },
    'take_profit': {
        'default': 0.03,  # 3%
        'extended': 0.05  # 5%
    }
}