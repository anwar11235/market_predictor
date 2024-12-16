"""
Market Integrations Module

This module handles market data integrations and real-time market data access:
- Finnhub market data
- Market indicators
- Real-time price data
"""

from .finnhub_market import FinnhubMarket

__all__ = [
    'FinnhubMarket',
    'get_market_client'
]

# Version of the market integrations module
__version__ = '0.1.0'

def get_market_client(provider: str, api_key: str, **kwargs):
    """
    Get market data client based on provider
    
    Args:
        provider: Market data provider ('finnhub')
        api_key: API key for the provider
        **kwargs: Additional provider-specific parameters
    
    Returns:
        Market data client instance
    
    Raises:
        ValueError: If provider is not supported
    """
    providers = {
        'finnhub': FinnhubMarket
    }
    
    if provider not in providers:
        raise ValueError(f"Unsupported provider. Available providers: {list(providers.keys())}")
    
    return providers[provider](api_key=api_key, **kwargs)

# Market data configurations
MARKET_PROVIDERS = {
    'finnhub': {
        'description': 'Real-time market data and basic fundamentals',
        'website': 'https://finnhub.io',
        'docs_url': 'https://finnhub.io/docs/api',
        'features': [
            'real_time_prices',
            'basic_financials',
            'company_news'
        ]
    }
}

# Market data types supported
MARKET_DATA_TYPES = {
    'price': ['open', 'high', 'low', 'close'],
    'volume': ['volume', 'vwap'],
    'indicators': ['sma', 'ema', 'rsi'],
    'fundamentals': ['market_cap', 'pe_ratio', 'eps']
}

# Default market settings
DEFAULT_SETTINGS = {
    'cache_expiry': 300,  # 5 minutes
    'rate_limit': 60,     # requests per minute
    'batch_size': 100     # symbols per batch
}