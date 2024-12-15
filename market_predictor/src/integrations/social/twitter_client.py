"""
Twitter Client Module

This module handles interactions with Twitter API to fetch and analyze:
- Cashtag mentions
- Financial discussions
- Market sentiment
- Trending topics in finance
"""

import tweepy
import pandas as pd
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from textblob import TextBlob

class TwitterClient:
    """Client for interacting with Twitter API"""
    
    # Financial accounts to track
    FINANCE_ACCOUNTS = [
        'DeItaone',      # Breaking news
        'CNBCnow',       # CNBC breaking news
        'MarketWatch',   # Market news
        'WSJmarkets',    # Wall Street Journal Markets
        'Stocktwits',    # StockTwits official
        'IBDinvestors'   # Investors Business Daily
    ]
    
    def __init__(self,
                 bearer_token: str,
                 api_key: str,
                 api_secret: str,
                 access_token: str,
                 access_secret: str,
                 cache_dir: Optional[str] = None):
        """
        Initialize Twitter client
        
        Args:
            bearer_token: Twitter API bearer token
            api_key: Twitter API key
            api_secret: Twitter API secret
            access_token: Twitter access token
            access_secret: Twitter access token secret
            cache_dir: Optional directory for caching responses
        """
        self.setup_logging()
        self.setup_cache(cache_dir)
        
        # Initialize Twitter API v2 client
        self.client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret,
            wait_on_rate_limit=True
        )
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TwitterClient')
    
    def setup_cache(self, cache_dir: Optional[str]):
        """Setup caching directory"""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def search_cashtag(self,
                      symbol: str,
                      hours: int = 24,
                      max_results: int = 100) -> pd.DataFrame:
        """
        Search for cashtag mentions
        
        Args:
            symbol: Stock symbol to search for
            hours: Number of hours to look back
            max_results: Maximum number of tweets to fetch
            
        Returns:
            DataFrame with tweet analysis
        """
        try:
            # Format cashtag query
            query = f"${symbol} -is:retweet lang:en"
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Search tweets
            tweets = self.client.search_recent_tweets(
                query=query,
                start_time=start_time,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return pd.DataFrame()
            
            # Process tweets
            processed_tweets = []
            for tweet in tweets.data:
                sentiment = self._analyze_text_sentiment(tweet.text)
                
                processed_tweets.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count'],
                    'sentiment_score': sentiment['sentiment_score'],
                    'sentiment_magnitude': sentiment['sentiment_magnitude']
                })
            
            return pd.DataFrame(processed_tweets)
            
        except Exception as e:
            self.logger.error(f"Error searching cashtag: {e}")
            return pd.DataFrame()
    
    def get_financial_tweets(self,
                           accounts: Optional[List[str]] = None,
                           hours: int = 24,
                           max_results: int = 100) -> pd.DataFrame:
        """
        Get tweets from financial accounts
        
        Args:
            accounts: List of Twitter accounts (default: FINANCE_ACCOUNTS)
            hours: Number of hours to look back
            max_results: Maximum tweets per account
            
        Returns:
            DataFrame with financial tweets
        """
        if accounts is None:
            accounts = self.FINANCE_ACCOUNTS
        
        all_tweets = []
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        try:
            for account in accounts:
                # Get user ID
                user = self.client.get_user(username=account)
                if not user.data:
                    continue
                
                # Get tweets
                tweets = self.client.get_users_tweets(
                    id=user.data.id,
                    start_time=start_time,
                    max_results=max_results,
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if not tweets.data:
                    continue
                
                # Process tweets
                for tweet in tweets.data:
                    sentiment = self._analyze_text_sentiment(tweet.text)
                    
                    all_tweets.append({
                        'account': account,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'likes': tweet.public_metrics['like_count'],
                        'retweets': tweet.public_metrics['retweet_count'],
                        'sentiment_score': sentiment['sentiment_score']
                    })
            
            return pd.DataFrame(all_tweets)
            
        except Exception as e:
            self.logger.error(f"Error fetching financial tweets: {e}")
            return pd.DataFrame()
    
    def get_market_sentiment(self,
                           symbol: str,
                           hours: int = 24) -> Dict[str, float]:
        """
        Get aggregated market sentiment for a symbol
        
        Args:
            symbol: Stock symbol
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            # Get tweets
            tweets_df = self.search_cashtag(symbol, hours=hours)
            
            if tweets_df.empty:
                return {
                    'average_sentiment': 0.0,
                    'sentiment_volume': 0,
                    'bullish_ratio': 0.0
                }
            
            # Calculate metrics
            avg_sentiment = tweets_df['sentiment_score'].mean()
            sentiment_volume = len(tweets_df)
            bullish_ratio = (tweets_df['sentiment_score'] > 0).mean()
            
            return {
                'average_sentiment': avg_sentiment,
                'sentiment_volume': sentiment_volume,
                'bullish_ratio': bullish_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market sentiment: {e}")
            return {
                'average_sentiment': 0.0,
                'sentiment_volume': 0,
                'bullish_ratio': 0.0
            }
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            analysis = TextBlob(text)
            return {
                'sentiment_score': analysis.sentiment.polarity,
                'sentiment_magnitude': abs(analysis.sentiment.polarity)
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_magnitude': 0.0
            }

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Initialize client
    client = TwitterClient(
        bearer_token=config.data.twitter_bearer_token,
        api_key=config.data.twitter_api_key,
        api_secret=config.data.twitter_api_secret,
        access_token=config.data.twitter_access_token,
        access_secret=config.data.twitter_access_secret,
        cache_dir='data/cache/twitter'
    )
    
    # Search for cashtag mentions
    tweets_df = client.search_cashtag('AAPL', hours=24)
    print("\nCashtag Mentions:")
    print(tweets_df.head())
    
    # Get financial tweets
    financial_df = client.get_financial_tweets(hours=24)
    print("\nFinancial Tweets:")
    print(financial_df.head())
    
    # Get market sentiment
    sentiment = client.get_market_sentiment('AAPL')
    print("\nMarket Sentiment:")
    print(json.dumps(sentiment, indent=2))