"""
News Integration Module

This module provides access to various news data sources:
- NewsAPI for general news
- Alpha Vantage for market news
- Finnhub for financial news
"""

from .newsapi_client import NewsAPIClient
from .alpha_vantage_news import AlphaVantageNewsClient
from .finnhub_client import FinnhubClient

__all__ = [
    'NewsAPIClient',
    'AlphaVantageNewsClient',
    'FinnhubClient',
    'get_news_client'
]

# Version of the news integration module
__version__ = '0.1.0'

def get_news_client(source: str, api_key: str, cache_dir: str = None):
    """
    Factory function to get appropriate news client
    
    Args:
        source: Name of news source ('newsapi', 'alphavantage', 'finnhub')
        api_key: API key for the selected service
        cache_dir: Optional directory for caching responses
        
    Returns:
        Initialized news client instance
        
    Raises:
        ValueError: If source is not supported
    """
    clients = {
        'newsapi': NewsAPIClient,
        'alphavantage': AlphaVantageNewsClient,
        'finnhub': FinnhubClient
    }
    
    if source not in clients:
        raise ValueError(f"Unsupported news source. Available sources: {list(clients.keys())}")
    
    return clients[source](api_key=api_key, cache_dir=cache_dir)

# Default news categories mapping
NEWS_CATEGORIES = {
    'market': ['stocks', 'commodities', 'forex', 'crypto'],
    'economy': ['macro', 'policy', 'central_banks'],
    'company': ['earnings', 'mergers', 'ipos']
}

# Supported news sources
NEWS_SOURCES = {
    'newsapi': {
        'description': 'General news with financial filters',
        'free_tier_limit': '100 requests/day'
    },
    'alphavantage': {
        'description': 'Market-focused news and sentiment',
        'free_tier_limit': '500 requests/day'
    },
    'finnhub': {
        'description': 'Real-time financial news',
        'free_tier_limit': '60 requests/minute'
    }
}