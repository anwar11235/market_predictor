"""
Finnhub API Client Module

Handles interactions with Finnhub API for:
- Market news
- Company news
- Basic sentiment data
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

class FinnhubClient:
    """Client for interacting with Finnhub API"""
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str, cache_dir: Optional[str] = None):
        """
        Initialize Finnhub client
        
        Args:
            api_key: Finnhub API key
            cache_dir: Optional directory for caching responses
        """
        self.api_key = api_key
        self.setup_logging()
        self.setup_cache(cache_dir)
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            'X-Finnhub-Token': self.api_key
        })
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FinnhubClient')
    
    def setup_cache(self, cache_dir: Optional[str]):
        """Setup caching directory"""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make API request
        
        Args:
            endpoint: API endpoint
            params: Optional query parameters
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Basic rate limiting
            time.sleep(0.1)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def get_market_news(self, category: str = 'general') -> pd.DataFrame:
        """
        Fetch market news
        
        Args:
            category: News category (general, forex, crypto, merger)
            
        Returns:
            DataFrame with news articles
        """
        try:
            response = self._make_request('/news', {'category': category})
            return self._process_news_articles(response)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market news: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def get_company_news(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetch company-specific news
        
        Args:
            symbol: Company symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with company news
        """
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date
        }
        
        try:
            response = self._make_request('/company-news', params)
            return self._process_news_articles(response)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch company news: {e}")
            return pd.DataFrame()
    
    def get_sentiment(self, symbol: str) -> Dict:
        """
        Get news sentiment for a symbol
        
        Args:
            symbol: Company symbol
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            return self._make_request('/news-sentiment', {'symbol': symbol})
            
        except Exception as e:
            self.logger.error(f"Failed to fetch sentiment: {e}")
            return {}
    
    def _process_news_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Process news articles into DataFrame
        
        Args:
            articles: List of news articles
        """
        if not articles:
            return pd.DataFrame()
        
        # Extract relevant fields
        processed_articles = []
        for article in articles:
            processed_articles.append({
                'id': article.get('id'),
                'title': article.get('headline'),
                'summary': article.get('summary'),
                'source': article.get('source'),
                'url': article.get('url'),
                'datetime': article.get('datetime'),
                'category': article.get('category'),
                'related': article.get('related', '')
            })
        
        # Create DataFrame
        df = pd.DataFrame(processed_articles)
        
        # Convert timestamp
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
            df.set_index('datetime', inplace=True)
        
        return df

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Initialize client
    client = FinnhubClient(
        api_key=config.data.finnhub_api_key,
        cache_dir='data/cache/finnhub'
    )
    
    # Example: Get market news
    market_news = client.get_market_news(category='general')
    print("\nMarket News Sample:")
    print(market_news.head())
    
    # Example: Get company news
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    company_news = client.get_company_news('AAPL', from_date, to_date)
    print("\nCompany News Sample:")
    print(company_news.head())
    
    # Example: Get sentiment
    sentiment = client.get_sentiment('AAPL')
    print("\nSentiment Data:")
    print(json.dumps(sentiment, indent=2))
