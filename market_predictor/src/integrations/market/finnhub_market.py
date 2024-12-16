"""
Finnhub Market Data Module

Handles market data integration with Finnhub API:
- Real-time market data
- Historical price data
- Market indicators
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import time
from pathlib import Path
import json

class FinnhubMarket:
    """Client for Finnhub market data API"""
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str, cache_dir: Optional[str] = None):
        """
        Initialize Finnhub market client
        
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
            'X-Finnhub-Token': self.api_key,
            'Content-Type': 'application/json'
        })
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FinnhubMarket')
    
    def setup_cache(self, cache_dir: Optional[str]):
        """Setup caching directory"""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make API request with rate limiting
        
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
    
    def get_market_data(self, 
                       symbol: str,
                       start_date: str,
                       end_date: str,
                       resolution: str = 'D') -> pd.DataFrame:
        """
        Get historical market data
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resolution: Data resolution (D=Day, W=Week, M=Month)
        """
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())
        
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'from': start_ts,
            'to': end_ts
        }
        
        try:
            response = self._make_request('/stock/candle', params)
            
            if response['s'] == 'no_data':
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': response['o'],
                'High': response['h'],
                'Low': response['l'],
                'Close': response['c'],
                'Volume': response['v']
            }, index=pd.to_datetime(response['t'], unit='s'))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote
        
        Args:
            symbol: Stock symbol
        """
        try:
            return self._make_request('/quote', {'symbol': symbol})
        except Exception as e:
            self.logger.error(f"Failed to fetch quote: {e}")
            return {}
    
    def get_basic_financials(self, symbol: str) -> Dict:
        """
        Get basic financial metrics
        
        Args:
            symbol: Stock symbol
        """
        try:
            return self._make_request('/stock/metric', {
                'symbol': symbol,
                'metric': 'all'
            })
        except Exception as e:
            self.logger.error(f"Failed to fetch financials: {e}")
            return {}
    
    def get_market_indices(self) -> pd.DataFrame:
        """Get major market indices data"""
        indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
        data = {}
        
        for index in indices:
            quote = self.get_quote(index)
            if quote:
                data[index] = quote
        
        return pd.DataFrame.from_dict(data, orient='index')
    
    def get_sector_performance(self) -> pd.DataFrame:
        """Get sector performance data"""
        sectors = [
            'XLK', 'XLF', 'XLV', 'XLE', 
            'XLI', 'XLY', 'XLP', 'XLB', 
            'XLU', 'XLRE'
        ]
        
        performance = {}
        for sector in sectors:
            quote = self.get_quote(sector)
            if quote:
                performance[sector] = quote
        
        return pd.DataFrame.from_dict(performance, orient='index')
    
    def get_market_status(self) -> Dict:
        """Get current market status and trading hours"""
        try:
            return self._make_request('/stock/market-status')
        except Exception as e:
            self.logger.error(f"Failed to fetch market status: {e}")
            return {}
    
    def get_earnings_calendar(self, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get earnings calendar
        
        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
        """
        params = {}
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
        
        try:
            response = self._make_request('/calendar/earnings', params)
            return pd.DataFrame(response.get('earningsCalendar', []))
        except Exception as e:
            self.logger.error(f"Failed to fetch earnings calendar: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    from config import load_validated_config
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Initialize client
    client = FinnhubMarket(
        api_key=config.data.finnhub_api_key,
        cache_dir='data/cache/finnhub'
    )
    
    # Get market data
    market_data = client.get_market_data(
        'AAPL',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    print("\nMarket Data Sample:")
    print(market_data.head())
    
    # Get current quote
    quote = client.get_quote('AAPL')
    print("\nCurrent Quote:")
    print(quote)
    
    # Get market indices
    indices = client.get_market_indices()
    print("\nMarket Indices:")
    print(indices)