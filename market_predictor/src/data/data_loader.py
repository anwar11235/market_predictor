"""
Data Loader Module

This module handles data collection from various sources:
- Market data (yfinance)
- Macroeconomic data (FRED)
- Market sentiment data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Union
import logging
import requests
from fredapi import Fred
from config import Config
import json
import time

class DataLoader:
    """Main data loader class that orchestrates data collection from all sources"""
    
    def __init__(self, config: Config):
        """
        Initialize DataLoader with configuration
        
        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        self.setup_logging()
        self.setup_cache_dir()
        
        # Initialize API clients if keys are provided
        self.fred_client = None
        if self.config.data.fred_api_key:
            self.fred_client = Fred(api_key=self.config.data.fred_api_key)
    
    def setup_logging(self):
        """Configure logging for the data loader"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataLoader')
    
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        self.cache_dir = Path(self.config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_market_data(self) -> pd.DataFrame:
        """
        Fetch market data from Yahoo Finance
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.cache_dir / f"market_data_{self.config.data.ticker}.parquet"
        
        # Try to load from cache first
        if self.config.data.cache_data and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                latest_date = df.index.max()
                
                # If cache is recent enough, return it
                if latest_date.date() >= (datetime.now() - timedelta(days=1)).date():
                    self.logger.info("Using cached market data")
                    return df
            except Exception as e:
                self.logger.warning(f"Error reading cache: {e}")
        
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(self.config.data.ticker)
            df = ticker.history(
                start=self.config.data.start_date,
                end=self.config.data.end_date,
                interval=self.config.data.data_frequency
            )
            
            # Basic validation
            if df.empty:
                raise ValueError("No data received from Yahoo Finance")
            
            # Save to cache if enabled
            if self.config.data.cache_data:
                df.to_parquet(cache_file)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            raise
    
    def get_macro_data(self) -> pd.DataFrame:
        """
        Fetch macroeconomic data from FRED
        
        Returns:
            DataFrame with macroeconomic indicators
        """
        if not self.fred_client:
            self.logger.warning("FRED API key not provided. Skipping macro data.")
            return pd.DataFrame()
        
        cache_file = self.cache_dir / "macro_data.parquet"
        
        # Try to load from cache first
        if self.config.data.cache_data and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                latest_date = df.index.max()
                
                # For macro data, we can use cache if it's less than a week old
                if latest_date.date() >= (datetime.now() - timedelta(days=7)).date():
                    self.logger.info("Using cached macro data")
                    return df
            except Exception as e:
                self.logger.warning(f"Error reading macro cache: {e}")
        
        try:
            # Initialize empty DataFrame for macro data
            macro_data = pd.DataFrame()
            
            # Fetch each macro indicator
            for series_id in self.config.macro.fred_series:
                series = self.fred_client.get_series(
                    series_id,
                    observation_start=self.config.data.start_date,
                    observation_end=self.config.data.end_date
                )
                macro_data[series_id] = series
            
            # Handle different frequencies and missing data
            macro_data = self._process_macro_data(macro_data)
            
            # Save to cache if enabled
            if self.config.data.cache_data:
                macro_data.to_parquet(cache_file)
            
            return macro_data
            
        except Exception as e:
            self.logger.error(f"Error fetching macro data: {e}")
            raise
    
    def _process_macro_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process macroeconomic data to align frequencies and handle missing values
        
        Args:
            df: Raw macro data DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Forward fill missing values up to specified limit
        df = df.resample('D').asfreq()
        
        if self.config.macro.interpolation_method == 'forward_fill':
            df = df.fillna(method='ffill', limit=30)  # Limit to 30 days
        elif self.config.macro.interpolation_method == 'linear':
            df = df.interpolate(method='linear', limit=30)
        
        return df
    
    def get_sentiment_data(self) -> pd.DataFrame:
        """
        Fetch market sentiment data from available free sources
        
        Returns:
            DataFrame with sentiment indicators
        """
        cache_file = self.cache_dir / "sentiment_data.parquet"
        
        # Try to load from cache first
        if self.config.data.cache_data and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                latest_date = df.index.max()
                
                # For sentiment data, use cache if it's from today
                if latest_date.date() >= datetime.now().date():
                    self.logger.info("Using cached sentiment data")
                    return df
            except Exception as e:
                self.logger.warning(f"Error reading sentiment cache: {e}")
        
        # Initialize empty DataFrame for sentiment data
        sentiment_data = pd.DataFrame()
        
        # Add basic sentiment indicators (Fear & Greed Index proxy)
        try:
            market_data = self.get_market_data()
            sentiment_data['volatility_sentiment'] = self._calculate_volatility_sentiment(market_data)
            sentiment_data['momentum_sentiment'] = self._calculate_momentum_sentiment(market_data)
            
            # Save to cache if enabled
            if self.config.data.cache_data:
                sentiment_data.to_parquet(cache_file)
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment data: {e}")
            raise
    
    def _calculate_volatility_sentiment(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate sentiment based on volatility"""
        returns = market_data['Close'].pct_change()
        volatility = returns.rolling(window=20).std()
        return -volatility  # Higher volatility = negative sentiment
    
    def _calculate_momentum_sentiment(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate sentiment based on momentum"""
        return market_data['Close'].pct_change(periods=10)  # 10-day momentum
    
    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available data and return as dictionary of DataFrames
        
        Returns:
            Dictionary containing market, macro, and sentiment DataFrames
        """
        data = {}
        
        # Fetch market data (required)
        data['market'] = self.get_market_data()
        
        # Fetch macro data if enabled
        if self.config.data.use_macro_data:
            data['macro'] = self.get_macro_data()
        
        # Fetch sentiment data if enabled
        if self.config.data.use_sentiment_data:
            data['sentiment'] = self.get_sentiment_data()
        
        return data

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Initialize data loader
    loader = DataLoader(config)
    
    # Fetch all data
    data = loader.get_all_data()
    
    # Print data info
    for name, df in data.items():
        print(f"\n{name.upper()} DATA:")
        print(df.info())