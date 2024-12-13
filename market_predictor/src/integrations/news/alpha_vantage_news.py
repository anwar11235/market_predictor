"""
Alpha Vantage News Client Module

This module handles interactions with Alpha Vantage News API:
- News and sentiment data
- Topic-based news filtering
- Tickers and market news
"""

import requests
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import json
from pathlib import Path

class AlphaVantageNewsClient:
    """Client for interacting with Alpha Vantage News API"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Available functions
    FUNCTIONS = {
        'news': 'NEWS_SENTIMENT',
        'ticker_news': 'NEWS_SENTIMENT'
    }
    
    # Topics available in Alpha Vantage
    TOPICS = [
        'technology', 'forex', 'merger_and_acquisitions',
        'financial_markets', 'economy_fiscal', 'economy_monetary',
        'economy_macro', 'energy_transportation', 'finance', 
        'life_sciences', 'manufacturing', 'real_estate'
    ]
    
    def __init__(self, api_key: str, cache_dir: Optional[str] = None):
        """
        Initialize Alpha Vantage news client
        
        Args:
            api_key: Alpha Vantage API key
            cache_dir: Optional directory for caching responses
        """
        self.api_key = api_key
        self.setup_logging()
        self.setup_cache(cache_dir)
        
        # Initialize session for connection pooling
        self.session = requests.Session()
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AlphaVantageNewsClient')
    
    def setup_cache(self, cache_dir: Optional[str]):
        """Setup caching directory"""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def _make_request(self, params: Dict) -> Dict:
        """
        Make API request with rate limiting
        
        Args:
            params: Query parameters
            
        Returns:
            JSON response
        """
        # Add API key to parameters
        params['apikey'] = self.api_key
        
        try:
            # Add delay to respect rate limits
            time.sleep(0.1)  # Basic rate limiting
            
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API error messages
            if 'Error Message' in data:
                raise ValueError(data['Error Message'])
                
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def get_market_news(self, 
                       topics: Optional[List[str]] = None,
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch market news and sentiment
        
        Args:
            topics: Optional list of topics to filter by
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with news and sentiment data
        """
        cache_file = None
        if self.cache_dir and use_cache:
            topic_str = '_'.join(topics) if topics else 'all'
            cache_file = self.cache_dir / f"av_market_news_{topic_str}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return pd.DataFrame(json.load(f))
        
        params = {
            'function': self.FUNCTIONS['news']
        }
        
        if topics:
            # Validate topics
            invalid_topics = [t for t in topics if t not in self.TOPICS]
            if invalid_topics:
                raise ValueError(f"Invalid topics: {invalid_topics}")
            params['topics'] = ','.join(topics)
        
        try:
            response = self._make_request(params)
            feed = response.get('feed', [])
            
            # Transform to DataFrame
            df = self._process_news_feed(feed)
            
            # Cache if enabled
            if cache_file:
                df.to_json(cache_file, orient='records')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market news: {e}")
            raise
    
    def get_ticker_news(self, 
                       tickers: Union[str, List[str]],
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch news for specific tickers
        
        Args:
            tickers: Single ticker or list of tickers
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with ticker-specific news
        """
        if isinstance(tickers, str):
            tickers = [tickers]
            
        cache_file = None
        if self.cache_dir and use_cache:
            tickers_str = '_'.join(tickers)
            cache_file = self.cache_dir / f"av_ticker_news_{tickers_str}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return pd.DataFrame(json.load(f))
        
        params = {
            'function': self.FUNCTIONS['ticker_news'],
            'tickers': ','.join(tickers)
        }
        
        try:
            response = self._make_request(params)
            feed = response.get('feed', [])
            
            df = self._process_news_feed(feed)
            
            if cache_file:
                df.to_json(cache_file, orient='records')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker news: {e}")
            raise
    
    def _process_news_feed(self, feed: List[Dict]) -> pd.DataFrame:
        """
        Process news feed into DataFrame
        
        Args:
            feed: List of news items
            
        Returns:
            Processed DataFrame
        """
        if not feed:
            return pd.DataFrame()
        
        processed_news = []
        for item in feed:
            processed_item = {
                'title': item.get('title'),
                'url': item.get('url'),
                'source': item.get('source'),
                'published_at': item.get('time_published'),
                'summary': item.get('summary'),
                'topics': ','.join(item.get('topics', [])),
                'sentiment_score': item.get('overall_sentiment_score'),
                'sentiment_label': item.get('overall_sentiment_label'),
                'relevance_score': item.get('relevance_score', 1.0)
            }
            
            # Add ticker-specific sentiment if available
            if 'ticker_sentiment' in item:
                ticker_sentiments = item['ticker_sentiment']
                for ticker_data in ticker_sentiments:
                    ticker = ticker_data.get('ticker')
                    processed_item[f'{ticker}_sentiment_score'] = ticker_data.get('sentiment_score')
                    processed_item[f'{ticker}_relevance_score'] = ticker_data.get('relevance_score')
            
            processed_news.append(processed_item)
        
        # Create DataFrame
        df = pd.DataFrame(processed_news)
        
        # Convert timestamp
        if 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df.set_index('published_at', inplace=True)
        
        return df
    
    def get_available_topics(self) -> List[str]:
        """
        Get list of available topics
        
        Returns:
            List of valid topics
        """
        return self.TOPICS.copy()

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Initialize client
    client = AlphaVantageNewsClient(
        api_key=config.data.alpha_vantage_api_key,
        cache_dir='data/cache/alpha_vantage'
    )
    
    # Get market news for specific topics
    topics = ['financial_markets', 'economy_macro']
    market_news = client.get_market_news(topics=topics)
    
    # Get news for specific tickers
    ticker_news = client.get_ticker_news(['AAPL', 'MSFT'])
    
    # Print sample
    print("\nMarket News Sample:")
    print(market_news.head())
    
    print("\nTicker News Sample:")
    print(ticker_news.head())