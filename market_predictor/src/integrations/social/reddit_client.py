"""
Reddit Client Module

This module handles interactions with Reddit API to fetch and analyze:
- Subreddit posts and comments
- Trading sentiment
- Stock mentions
- Popular discussions
"""

import praw
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from textblob import TextBlob

class RedditClient:
    """Client for interacting with Reddit API"""
    
    # Finance-related subreddits
    FINANCE_SUBREDDITS = [
        'wallstreetbets',
        'stocks',
        'investing',
        'stockmarket',
        'options',
        'SecurityAnalysis'
    ]
    
    def __init__(self, 
                 client_id: str,
                 client_secret: str,
                 user_agent: str,
                 cache_dir: Optional[str] = None):
        """
        Initialize Reddit client
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: Reddit API user agent
            cache_dir: Optional directory for caching responses
        """
        self.setup_logging()
        self.setup_cache(cache_dir)
        
        # Initialize PRAW
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('RedditClient')
    
    def setup_cache(self, cache_dir: Optional[str]):
        """Setup caching directory"""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def get_subreddit_sentiment(self,
                              subreddit_name: str,
                              time_filter: str = 'day',
                              limit: int = 100) -> pd.DataFrame:
        """
        Get sentiment analysis of subreddit posts
        
        Args:
            subreddit_name: Name of subreddit
            time_filter: One of (hour, day, week, month, year, all)
            limit: Maximum number of posts to analyze
            
        Returns:
            DataFrame with post sentiment analysis
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                sentiment = self._analyze_text_sentiment(post.title + " " + post.selftext)
                
                posts.append({
                    'id': post.id,
                    'title': post.title,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'upvote_ratio': post.upvote_ratio,
                    'sentiment_score': sentiment['sentiment_score'],
                    'sentiment_magnitude': sentiment['sentiment_magnitude']
                })
            
            return pd.DataFrame(posts)
            
        except Exception as e:
            self.logger.error(f"Error fetching subreddit sentiment: {e}")
            return pd.DataFrame()
    
    def get_stock_mentions(self,
                          symbol: str,
                          subreddits: Optional[List[str]] = None,
                          days: int = 1) -> pd.DataFrame:
        """
        Get mentions of a stock symbol across subreddits
        
        Args:
            symbol: Stock symbol to search for
            subreddits: List of subreddits to search (default: FINANCE_SUBREDDITS)
            days: Number of days to look back
            
        Returns:
            DataFrame with stock mentions and analysis
        """
        if subreddits is None:
            subreddits = self.FINANCE_SUBREDDITS
            
        mentions = []
        
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for symbol in posts
                for post in subreddit.search(symbol, time_filter='day', limit=100):
                    if (datetime.utcnow() - datetime.fromtimestamp(post.created_utc)) <= timedelta(days=days):
                        sentiment = self._analyze_text_sentiment(post.title + " " + post.selftext)
                        
                        mentions.append({
                            'subreddit': subreddit_name,
                            'title': post.title,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'sentiment_score': sentiment['sentiment_score'],
                            'sentiment_magnitude': sentiment['sentiment_magnitude']
                        })
            
            return pd.DataFrame(mentions)
            
        except Exception as e:
            self.logger.error(f"Error fetching stock mentions: {e}")
            return pd.DataFrame()
    
    def get_trending_discussion(self,
                              subreddit_name: str,
                              min_score: int = 10,
                              time_filter: str = 'day') -> pd.DataFrame:
        """
        Get trending discussions from a subreddit
        
        Args:
            subreddit_name: Name of subreddit
            min_score: Minimum score threshold
            time_filter: One of (hour, day, week, month, year, all)
            
        Returns:
            DataFrame with trending discussions
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            discussions = []
            
            for post in subreddit.hot(limit=50):
                if post.score >= min_score:
                    sentiment = self._analyze_text_sentiment(post.title + " " + post.selftext)
                    
                    discussions.append({
                        'title': post.title,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'upvote_ratio': post.upvote_ratio,
                        'sentiment_score': sentiment['sentiment_score'],
                        'url': post.url
                    })
            
            return pd.DataFrame(discussions)
            
        except Exception as e:
            self.logger.error(f"Error fetching trending discussions: {e}")
            return pd.DataFrame()
    
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
    client = RedditClient(
        client_id=config.data.reddit_client_id,
        client_secret=config.data.reddit_client_secret,
        user_agent='MarketPredictor/1.0',
        cache_dir='data/cache/reddit'
    )
    
    # Get sentiment from r/stocks
    sentiment_df = client.get_subreddit_sentiment('stocks')
    print("\nSubreddit Sentiment:")
    print(sentiment_df.head())
    
    # Get mentions of a stock
    mentions_df = client.get_stock_mentions('AAPL')
    print("\nStock Mentions:")
    print(mentions_df.head())
    
    # Get trending discussions
    trending_df = client.get_trending_discussion('wallstreetbets')
    print("\nTrending Discussions:")
    print(trending_df.head())