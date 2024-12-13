"""
NewsAPI Client Module

This module handles interactions with NewsAPI to fetch financial news:
- Market news
- Company-specific news
- Economic news
- Sentiment analysis
"""

import requests
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from ratelimit import limits, sleep_and_retry
import json
from pathlib import Path

class NewsAPIClient:
    """Client for interacting with NewsAPI"""
    
    # NewsAPI base URL
    BASE_URL = "https://newsapi.org/v2"
    
    # Endpoints
    ENDPOINTS = {
        'everything': '/everything',
        'top_headlines': '/top-headlines'
    }
    
    # Rate limits (free tier)
    CALLS_PER_DAY = 100
    CALLS_PER_MINUTE = 10
    
    def __init__(self, api_key: str, cache_dir: Optional[str] = None):
        """
        Initialize NewsAPI client
        
        Args:
            api_key: NewsAPI authentication key
            cache_dir: Optional directory for caching responses
        """
        self.api_key = api_key
        self.setup_logging()
        self.setup_cache(cache_dir)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'MarketPredictor/1.0'
        })
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('NewsAPIClient')
    
    def setup_cache(self, cache_dir: Optional[str]):
        """Setup caching directory"""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    @sleep_and_retry
    @limits(calls=CALLS_PER_MINUTE, period=60)
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make rate-limited API request
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            
        Returns:
            JSON response
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def get_market_news(self,
                       days_back: int = 7,
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch market-related news
        
        Args:
            days_back: Number of days to look back
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with news articles
        """
        cache_file = None
        if self.cache_dir and use_cache:
            cache_file = self.cache_dir / f"market_news_{days_back}days.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return pd.DataFrame(json.load(f))
        
        # Prepare query parameters
        params = {
            'q': 'stock market OR financial markets OR SP500',
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': (datetime.now() - timedelta(days=days_back)).isoformat(),
            'to': datetime.now().isoformat()
        }
        
        try:
            response = self._make_request(self.ENDPOINTS['everything'], params)
            articles = response.get('articles', [])
            
            # Transform to DataFrame
            df = self._process_articles(articles)
            
            # Cache if enabled
            if cache_file:
                df.to_json(cache_file, orient='records')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market news: {e}")
            raise
    
    def get_company_news(self,
                        company: str,
                        days_back: int = 7) -> pd.DataFrame:
        """
        Fetch company-specific news
        
        Args:
            company: Company name or ticker
            days_back: Number of days to look back
            
        Returns:
            DataFrame with company news
        """
        params = {
            'q': f'"{company}" stock OR "{company}" shares',
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': (datetime.now() - timedelta(days=days_back)).isoformat(),
            'to': datetime.now().isoformat()
        }
        
        try:
            response = self._make_request(self.ENDPOINTS['everything'], params)
            articles = response.get('articles', [])
            return self._process_articles(articles)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch company news: {e}")
            raise
    
    def _process_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Process raw articles into DataFrame
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Processed DataFrame
        """
        if not articles:
            return pd.DataFrame()
        
        # Extract relevant fields
        processed_articles = []
        for article in articles:
            processed_articles.append({
                'title': article.get('title'),
                'description': article.get('description'),
                'source': article.get('source', {}).get('name'),
                'url': article.get('url'),
                'published_at': article.get('publishedAt'),
                'content': article.get('content')
            })
        
        # Create DataFrame
        df = pd.DataFrame(processed_articles)
        
        # Convert timestamp
        if 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df.set_index('published_at', inplace=True)
        
        return df
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text (placeholder for actual sentiment analysis)
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        # This is a placeholder. In practice, you'd want to use a proper
        # sentiment analysis model or service
        from textblob import TextBlob
        
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def get_news_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment analysis to news DataFrame
        
        Args:
            df: DataFrame with news articles
            
        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty:
            return df
        
        sentiments = []
        for _, row in df.iterrows():
            # Analyze both title and description
            title_sentiment = self.analyze_sentiment(row['title']) if row['title'] else {'polarity': 0, 'subjectivity': 0}
            desc_sentiment = self.analyze_sentiment(row['description']) if row['description'] else {'polarity': 0, 'subjectivity': 0}
            
            # Average the sentiments
            sentiments.append({
                'sentiment_polarity': (title_sentiment['polarity'] + desc_sentiment['polarity']) / 2,
                'sentiment_subjectivity': (title_sentiment['subjectivity'] + desc_sentiment['subjectivity']) / 2
            })
        
        sentiment_df = pd.DataFrame(sentiments)
        return pd.concat([df, sentiment_df], axis=1)

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Initialize client
    client = NewsAPIClient(
        api_key=config.data.news_api_key,
        cache_dir='data/cache/news'
    )
    
    # Fetch market news
    market_news = client.get_market_news(days_back=7)
    
    # Add sentiment analysis
    market_news_with_sentiment = client.get_news_sentiment(market_news)
    
    # Print sample
    print("\nRecent Market News with Sentiment:")
    print(market_news_with_sentiment.head())