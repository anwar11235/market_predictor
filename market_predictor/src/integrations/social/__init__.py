"""
Social Media Integration Module

This module provides access to social media data sources:
- Reddit for trading communities
- Twitter for cashtag analysis and financial news
"""

from .reddit_client import RedditClient
from .twitter_client import TwitterClient

__all__ = [
    'RedditClient',
    'TwitterClient',
    'get_social_client',
    'get_social_sentiment'
]

# Version of the social integration module
__version__ = '0.1.0'

def get_social_client(platform: str, credentials: dict, cache_dir: str = None):
    """
    Factory function to get appropriate social media client
    
    Args:
        platform: Platform name ('reddit' or 'twitter')
        credentials: Dictionary containing platform-specific credentials
        cache_dir: Optional directory for caching responses
        
    Returns:
        Initialized social media client
        
    Raises:
        ValueError: If platform is not supported
    """
    if platform == 'reddit':
        return RedditClient(
            client_id=credentials.get('client_id'),
            client_secret=credentials.get('client_secret'),
            user_agent=credentials.get('user_agent', 'MarketPredictor/1.0'),
            cache_dir=cache_dir
        )
    elif platform == 'twitter':
        return TwitterClient(
            bearer_token=credentials.get('bearer_token'),
            api_key=credentials.get('api_key'),
            api_secret=credentials.get('api_secret'),
            access_token=credentials.get('access_token'),
            access_secret=credentials.get('access_secret'),
            cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unsupported platform: {platform}. Use 'reddit' or 'twitter'")

def get_social_sentiment(symbol: str, clients: dict) -> dict:
    """
    Get aggregated sentiment from all social sources
    
    Args:
        symbol: Stock symbol to analyze
        clients: Dictionary of initialized clients
        
    Returns:
        Dictionary containing sentiment metrics from each platform
    """
    sentiment_data = {}
    
    # Get Reddit sentiment
    if 'reddit' in clients:
        try:
            mentions = clients['reddit'].get_stock_mentions(symbol)
            if not mentions.empty:
                sentiment_data['reddit'] = {
                    'mention_count': len(mentions),
                    'avg_sentiment': mentions['sentiment_score'].mean(),
                    'total_score': mentions['score'].sum()
                }
        except Exception as e:
            sentiment_data['reddit'] = {'error': str(e)}
    
    # Get Twitter sentiment
    if 'twitter' in clients:
        try:
            twitter_sentiment = clients['twitter'].get_market_sentiment(symbol)
            sentiment_data['twitter'] = twitter_sentiment
        except Exception as e:
            sentiment_data['twitter'] = {'error': str(e)}
    
    return sentiment_data

# Platform-specific configurations
PLATFORM_CONFIG = {
    'reddit': {
        'required_credentials': ['client_id', 'client_secret'],
        'default_subreddits': [
            'wallstreetbets',
            'stocks',
            'investing',
            'stockmarket'
        ],
        'rate_limits': 'OAuth2 limits apply'
    },
    'twitter': {
        'required_credentials': [
            'bearer_token',
            'api_key',
            'api_secret',
            'access_token',
            'access_secret'
        ],
        'default_accounts': [
            'DeItaone',
            'CNBCnow',
            'MarketWatch',
            'WSJmarkets'
        ],
        'rate_limits': '500k tweets/month'
    }
}

# Common financial topics for tracking
FINANCIAL_TOPICS = {
    'market_sentiment': ['bullish', 'bearish', 'neutral'],
    'market_events': ['earnings', 'ipo', 'merger', 'acquisition'],
    'trading_signals': ['buy', 'sell', 'hold'],
    'risk_factors': ['volatility', 'uncertainty', 'risk']
}