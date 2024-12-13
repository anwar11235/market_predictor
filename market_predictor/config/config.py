from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
from pathlib import Path

@dataclass
class DataConfig:
    """Configuration for data loading and processing"""
    # Market Data Settings
    ticker: str = '^GSPC'  # S&P 500 index
    start_date: str = '2010-01-01'
    end_date: Optional[str] = None  # None means up to current date
    training_start: str = '2010-01-01'
    validation_start: str = '2018-01-01'
    test_start: str = '2020-01-01'
    data_frequency: str = 'daily'
    adjust_prices: bool = True
    
    # Data Sources
    use_market_data: bool = True
    use_macro_data: bool = True
    use_sentiment_data: bool = True
    use_alternative_data: bool = True
    
    # API Keys and Access Settings
    fred_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    twitter_api_key: Optional[str] = None
    
    # General Settings
    random_seed: int = 42
    cache_data: bool = True
    cache_dir: str = './data/cache'

@dataclass
class TechnicalFeatureConfig:
    """Configuration for technical analysis features"""
    # Return-based features
    return_types: List[str] = field(default_factory=lambda: ['Returns', 'Log_Returns'])
    
    # Moving Average features
    ma_windows: List[int] = field(default_factory=lambda: [5, 20, 50])
    price_distance_windows: List[int] = field(default_factory=lambda: [5, 20, 50])
    
    # Volume-based features
    volume_features: List[str] = field(default_factory=lambda: [
        'OBV', 'Volume_Ratio', 'Force_Index_EMA13', 
        'Vol_Trend', 'PV_Trend', 'PV_Divergence', 'Volume_Force'
    ])
    
    # Volatility and Momentum
    volatility_windows: List[int] = field(default_factory=lambda: [20])
    rsi_windows: List[int] = field(default_factory=lambda: [7, 14, 28])
    atr_period: int = 14
    mfi_period: int = 14
    
    # Pattern and Price Action
    pattern_indicators: List[str] = field(default_factory=lambda: [
        'Higher_High', 'Lower_Low', 'Inside_Day'
    ])
    price_features: List[str] = field(default_factory=lambda: [
        'VWAP_Ratio', 'High_Low_Range', 'Daily_Gap',
        'New_Highs', 'New_Lows'
    ])
    trend_indicators: List[str] = field(default_factory=lambda: [
        'Bullish_Momentum', 'Bearish_Momentum', 'Trend_Strength'
    ])

@dataclass
class MacroFeatureConfig:
    """Configuration for macroeconomic features"""
    # Economic Indicators
    fred_series: List[str] = field(default_factory=lambda: [
        'GDP', 'UNRATE', 'CPIAUCSL', 'INDPRO', 'M2', 
        'DFF', 'T10Y2Y', 'PSAVERT', 'PCE'
    ])
    
    # Market Environment
    market_indicators: List[str] = field(default_factory=lambda: [
        'MOVE_Index', 'VIX', 'TED_Spread', 'HY_Spread'
    ])
    
    # Monetary Policy
    monetary_indicators: List[str] = field(default_factory=lambda: [
        'Fed_Funds_Rate', 'Fed_Balance_Sheet', 'M2_Growth'
    ])
    
    # Update Frequency Settings
    macro_update_frequency: str = 'monthly'
    interpolation_method: str = 'forward_fill'
    lag_adjustment: int = 1  # Months of lag for data availability

@dataclass
class SentimentFeatureConfig:
    """Configuration for sentiment analysis features"""
    # News Sentiment
    news_sources: List[str] = field(default_factory=lambda: [
        'reuters', 'bloomberg', 'wsj', 'ft'
    ])
    news_categories: List[str] = field(default_factory=lambda: [
        'economy', 'markets', 'policy'
    ])
    
    # Social Media Sentiment
    social_sources: List[str] = field(default_factory=lambda: [
        'twitter', 'reddit', 'stocktwits'
    ])
    
    # Market Sentiment
    market_sentiment_indicators: List[str] = field(default_factory=lambda: [
        'put_call_ratio', 'fear_greed_index', 'aaii_sentiment'
    ])
    
    # Processing Settings
    sentiment_windows: List[int] = field(default_factory=lambda: [1, 3, 7, 14])
    sentiment_aggregation: str = 'weighted_average'

@dataclass
class ModelConfig:
    """Configuration for model training and prediction"""
    # Basic Settings
    target_horizon: int = 1
    cv_folds: int = 5
    batch_size: int = 32
    max_epochs: int = 100
    
    # Training Settings
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_size: float = 0.2
    
    # Ensemble Settings
    ensemble_models: List[str] = field(default_factory=lambda: [
        'technical_model', 'macro_model', 'sentiment_model'
    ])
    ensemble_method: str = 'weighted_voting'
    dynamic_weights: bool = True
    
    # Agent Settings
    use_agents: bool = False  # For future agent-based implementation
    agent_update_frequency: str = 'daily'
    agent_memory_length: int = 100

class Config:
    """Main configuration class"""
    def __init__(self, config_path: Optional[str] = None):
        self.data = DataConfig()
        self.technical = TechnicalFeatureConfig()
        self.macro = MacroFeatureConfig()
        self.sentiment = SentimentFeatureConfig()
        self.model = ModelConfig()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        if 'data' in config_dict:
            self.data = DataConfig(**config_dict['data'])
        if 'technical' in config_dict:
            self.technical = TechnicalFeatureConfig(**config_dict['technical'])
        if 'macro' in config_dict:
            self.macro = MacroFeatureConfig(**config_dict['macro'])
        if 'sentiment' in config_dict:
            self.sentiment = SentimentFeatureConfig(**config_dict['sentiment'])
        if 'model' in config_dict:
            self.model = ModelConfig(**config_dict['model'])
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to YAML file"""
        config_dict = {
            'data': {k: v for k, v in self.data.__dict__.items()},
            'technical': {k: v for k, v in self.technical.__dict__.items()},
            'macro': {k: v for k, v in self.macro.__dict__.items()},
            'sentiment': {k: v for k, v in self.sentiment.__dict__.items()},
            'model': {k: v for k, v in self.model.__dict__.items()}
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def get_all_features(self) -> Dict[str, List[str]]:
        """Returns all features grouped by category"""
        features = {
            'technical': self.technical.return_types + 
                        [f'MA_{w}' for w in self.technical.ma_windows] +
                        self.technical.volume_features +
                        [f'RSI_{w}' for w in self.technical.rsi_windows] +
                        self.technical.pattern_indicators +
                        self.technical.price_features +
                        self.technical.trend_indicators,
            'macro': self.macro.fred_series +
                     self.macro.market_indicators +
                     self.macro.monetary_indicators,
            'sentiment': [f"{source}_{cat}" for source in self.sentiment.news_sources 
                         for cat in self.sentiment.news_categories] +
                        self.sentiment.market_sentiment_indicators
        }
        return features

if __name__ == "__main__":
    # Create default config
    config = Config()
    
    # Save default config
    config.save_config('config/parameters.yaml')
    
    # Example: Print all features
    all_features = config.get_all_features()
    for category, feature_list in all_features.items():
        print(f"\n{category.upper()} FEATURES:")
        for feature in feature_list:
            print(f"- {feature}")