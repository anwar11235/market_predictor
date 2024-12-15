"""
Market Predictor

A modular system for market prediction incorporating:
- Multiple data sources (market, news, sentiment)
- Feature engineering
- ML/DL models
- Trading strategies
"""

from .data import DataLoader
from .features import (
    TechnicalFeatures,
    SentimentFeatures,
    MacroFeatures,
    FeatureGenerator
)
from .models import (
    ModelFactory,
    EnsembleModel,
    create_model,
    create_ensemble
)
from .integrations import (
    create_data_clients,
    get_all_sentiment
)
from .utils import (
    setup_project_logger,
    ModelMetrics,
    TradingMetrics,
    RiskMetrics
)

__version__ = '0.1.0'

__all__ = [
    # Data handling
    'DataLoader',
    
    # Feature engineering
    'TechnicalFeatures',
    'SentimentFeatures',
    'MacroFeatures',
    'FeatureGenerator',
    
    # Models
    'ModelFactory',
    'EnsembleModel',
    'create_model',
    'create_ensemble',
    
    # Integrations
    'create_data_clients',
    'get_all_sentiment',
    
    # Utilities
    'setup_project_logger',
    'ModelMetrics',
    'TradingMetrics',
    'RiskMetrics',
    
    # Convenience functions
    'create_prediction_pipeline',
    'backtest_strategy'
]

def create_prediction_pipeline(config):
    """
    Create a complete prediction pipeline
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (data_loader, feature_generator, model)
    """
    # Initialize components
    data_loader = DataLoader(config)
    feature_gen = FeatureGenerator(config)
    model = create_ensemble(config, ['technical', 'macro', 'sentiment'])
    
    return data_loader, feature_gen, model

def backtest_strategy(config, model, data, start_date=None, end_date=None):
    """
    Backtest a prediction strategy
    
    Args:
        config: Configuration object
        model: Trained model
        data: Historical data
        start_date: Optional backtest start date
        end_date: Optional backtest end date
        
    Returns:
        Dictionary containing backtest results and metrics
    """
    logger = setup_project_logger('backtest')
    
    try:
        # Slice data if dates provided
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # Generate predictions
        predictions = model.predict(data)
        
        # Calculate metrics
        metrics = {
            'model_metrics': ModelMetrics.classification_metrics(
                data['target'],
                predictions
            ),
            'trading_metrics': {
                'sharpe': TradingMetrics.calculate_sharpe_ratio(data['returns']),
                'max_drawdown': TradingMetrics.calculate_max_drawdown(data['close']),
                'win_rate': TradingMetrics.calculate_win_rate(data['returns'])
            },
            'risk_metrics': {
                'var_95': RiskMetrics.calculate_var(data['returns']),
                'cvar_95': RiskMetrics.calculate_cvar(data['returns'])
            }
        }
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'data': data
        }
        
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        raise

# Version information
VERSION_INFO = {
    'major': 0,
    'minor': 1,
    'patch': 0,
    'release': 'alpha'
}

# Project metadata
PROJECT_METADATA = {
    'name': 'Market Predictor',
    'version': __version__,
    'description': 'Machine learning based market prediction system',
    'author': 'Your Name',
    'license': 'MIT',
    'python_requires': '>=3.8'
}