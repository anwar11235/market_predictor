"""
Market Predictor Integrations Module

This module provides unified access to external data sources:
- News APIs (NewsAPI, Alpha Vantage, Finnhub)
- Social Media (Reddit, Twitter)
- Market Data (Finnhub Market Data)
"""

from .news import NewsAPIClient, AlphaVantageNewsClient, FinnhubClient, get_news_client
from .social import RedditClient, TwitterClient, get_social_client, get_social_sentiment
from .market import FinnhubMarket

__all__ = [
    # News clients
    'NewsAPIClient',
    'AlphaVantageNewsClient',
    'FinnhubClient',
    'get_news_client',
    
    # Social clients
    'RedditClient',
    'TwitterClient',
    'get_social_client',
    'get_social_sentiment',
    
    # Market clients
    'FinnhubMarket',
    
    # Utility functions
    'create_data_clients',
    'get_all_sentiment'
]

# Version of the integrations module
__version__ = '0.1.0'

def create_data_clients(config) -> dict:
    """
    Create all necessary data clients based on configuration
    
    Args:
        config: Configuration object containing API credentials
        
    Returns:
        Dictionary containing initialized clients
    """
    clients = {}
    
    # Initialize news clients
    if config.data.use_news_data:
        try:
            if config.data.newsapi_key:
                clients['newsapi'] = get_news_client('newsapi', config.data.newsapi_key)
            if config.data.alpha_vantage_key:
                clients['alphavantage'] = get_news_client('alphavantage', config.data.alpha_vantage_key)
            if config.data.finnhub_key:
                clients['finnhub'] = get_news_client('finnhub', config.data.finnhub_key)
        except Exception as e:
            print(f"Error initializing news clients: {e}")
    
    # Initialize social clients
    if config.data.use_social_data:
        try:
            # Reddit setup
            if config.data.reddit_client_id:
                reddit_credentials = {
                    'client_id': config.data.reddit_client_id,
                    'client_secret': config.data.reddit_client_secret,
                    'user_agent': 'MarketPredictor/1.0'
                }
                clients['reddit'] = get_social_client('reddit', reddit_credentials)
            
            # Twitter setup
            if config.data.twitter_bearer_token:
                twitter_credentials = {
                    'bearer_token': config.data.twitter_bearer_token,
                    'api_key': config.data.twitter_api_key,
                    'api_secret': config.data.twitter_api_secret,
                    'access_token': config.data.twitter_access_token,
                    'access_secret': config.data.twitter_access_secret
                }
                clients['twitter'] = get_social_client('twitter', twitter_credentials)
        except Exception as e:
            print(f"Error initializing social clients: {e}")
    
    # Initialize market data client
    if config.data.use_market_data and config.data.finnhub_key:
        try:
            clients['market'] = FinnhubMarket(config.data.finnhub_key)
        except Exception as e:
            print(f"Error initializing market client: {e}")
    
    return clients

def get_all_sentiment(symbol: str, clients: dict) -> dict:
    """
    Get aggregated sentiment from all available sources
    
    Args:
        symbol: Stock symbol to analyze
        clients: Dictionary of initialized clients
        
    Returns:
        Dictionary containing sentiment from all sources
    """
    sentiment_data = {
        'news': {},
        'social': {},
        'market': {}
    }
    
    # Get news sentiment
    news_clients = {k: v for k, v in clients.items() if k in ['newsapi', 'alphavantage', 'finnhub']}
    for source, client in news_clients.items():
        try:
            if hasattr(client, 'get_sentiment'):
                sentiment_data['news'][source] = client.get_sentiment(symbol)
        except Exception as e:
            sentiment_data['news'][source] = {'error': str(e)}
    
    # Get social sentiment
    social_clients = {k: v for k, v in clients.items() if k in ['reddit', 'twitter']}
    sentiment_data['social'] = get_social_sentiment(symbol, social_clients)
    
    # Get market sentiment if available
    if 'market' in clients:
        try:
            sentiment_data['market'] = clients['market'].get_market_sentiment(symbol)
        except Exception as e:
            sentiment_data['market'] = {'error': str(e)}
    
    return sentiment_data

# Data source configurations
DATA_SOURCES = {
    'news': {
        'newsapi': {'type': 'general_news', 'update_frequency': 'real-time'},
        'alphavantage': {'type': 'market_news', 'update_frequency': '24h'},
        'finnhub': {'type': 'financial_news', 'update_frequency': 'real-time'}
    },
    'social': {
        'reddit': {'type': 'community_sentiment', 'update_frequency': 'real-time'},
        'twitter': {'type': 'social_sentiment', 'update_frequency': 'real-time'}
    },
    'market': {
        'finnhub': {'type': 'market_data', 'update_frequency': 'real-time'}
    }
}