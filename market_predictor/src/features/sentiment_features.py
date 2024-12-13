"""
Sentiment Features Module

This module handles sentiment analysis from multiple sources:
- Market sentiment indicators (Fear & Greed proxy)
- News sentiment analysis using free APIs
- Technical-based sentiment indicators
- Social media sentiment proxies
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from config import Config
import requests
from datetime import datetime, timedelta
import json
import re
from textblob import TextBlob
import newspaper
from bs4 import BeautifulSoup

class SentimentFeatures:
    """Handles calculation and aggregation of sentiment features"""
    
    def __init__(self, config: Config):
        """
        Initialize SentimentFeatures with configuration
        
        Args:
            config: Configuration object containing sentiment parameters
        """
        self.config = config
        self.setup_logging()
        self.cache = {}
    
    def setup_logging(self):
        """Configure logging for the sentiment analyzer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SentimentFeatures')
    
    def calculate_all_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all configured sentiment features
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all sentiment features
        """
        features = pd.DataFrame(index=market_data.index)
        
        try:
            # Calculate market-based sentiment indicators
            self._add_market_sentiment_features(market_data, features)
            
            # Add news sentiment if API key is available
            if self.config.data.news_api_key:
                self._add_news_sentiment_features(features)
            
            # Add technical-based sentiment indicators
            self._add_technical_sentiment_features(market_data, features)
            
            # Add aggregated sentiment scores
            self._add_aggregated_sentiment(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment features: {e}")
            raise
    
    def _add_market_sentiment_features(self, market_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate market-based sentiment indicators"""
        # Put-Call Ratio Proxy (using volatility)
        features['Put_Call_Proxy'] = self._calculate_put_call_proxy(market_data)
        
        # Fear & Greed Index Components
        features['Market_Momentum'] = self._calculate_market_momentum(market_data)
        features['Market_Volatility'] = self._calculate_market_volatility(market_data)
        features['Market_Strength'] = self._calculate_market_strength(market_data)
        
        # Aggregate Fear & Greed Score (0-100)
        features['Fear_Greed_Index'] = self._calculate_fear_greed_index(features)
    
    def _calculate_put_call_proxy(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate put-call ratio proxy using volatility and price action"""
        returns = market_data['Close'].pct_change()
        volatility = returns.rolling(window=20).std()
        price_trend = returns.rolling(window=10).mean()
        
        # Higher values indicate more puts (fear), lower values indicate more calls (greed)
        put_call_proxy = (volatility * 100) * (1 - price_trend)
        return put_call_proxy.rolling(window=5).mean()
    
    def _calculate_market_momentum(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate market momentum indicator"""
        close_prices = market_data['Close']
        
        # Compare current price to moving averages
        ma_10 = close_prices.rolling(window=10).mean()
        ma_30 = close_prices.rolling(window=30).mean()
        
        momentum = ((close_prices - ma_10) / ma_10) * 0.5 + \
                  ((close_prices - ma_30) / ma_30) * 0.5
        
        # Scale to 0-100
        return self._scale_to_100(momentum)
    
    def _calculate_market_volatility(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate market volatility indicator"""
        returns = market_data['Close'].pct_change()
        current_vol = returns.rolling(window=20).std()
        historical_vol = returns.rolling(window=100).std()
        
        # Compare current volatility to historical
        rel_volatility = (current_vol / historical_vol)
        
        # Inverse scale (higher volatility = more fear)
        return 100 - self._scale_to_100(rel_volatility)
    
    def _calculate_market_strength(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate market strength indicator"""
        # Use price momentum and volume
        returns = market_data['Close'].pct_change()
        volume_change = market_data['Volume'].pct_change()
        
        # Combine price and volume momentum
        strength = returns.rolling(window=10).mean() * \
                  volume_change.rolling(window=10).mean()
        
        return self._scale_to_100(strength)
    
    def _calculate_fear_greed_index(self, features: pd.DataFrame) -> pd.Series:
        """Calculate aggregated fear and greed index"""
        components = [
            'Market_Momentum',
            'Market_Volatility',
            'Market_Strength'
        ]
        
        # Equal-weighted average of components
        fear_greed = features[components].mean(axis=1)
        
        # Ensure values are between 0-100
        return fear_greed.clip(0, 100)
    
    def _add_technical_sentiment_features(self, market_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate technical-based sentiment indicators"""
        # RSI-based sentiment
        rsi = self._calculate_rsi(market_data['Close'], 14)
        features['RSI_Sentiment'] = self._scale_to_100(rsi)
        
        # MACD-based sentiment
        macd_sentiment = self._calculate_macd_sentiment(market_data['Close'])
        features['MACD_Sentiment'] = self._scale_to_100(macd_sentiment)
        
        # Volume-based sentiment
        vol_sentiment = self._calculate_volume_sentiment(market_data)
        features['Volume_Sentiment'] = self._scale_to_100(vol_sentiment)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd_sentiment(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD-based sentiment"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal
    
    def _calculate_volume_sentiment(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate volume-based sentiment"""
        volume = market_data['Volume']
        price_change = market_data['Close'].pct_change()
        
        # Positive volume sentiment when price increase comes with volume increase
        return volume.pct_change() * price_change
    
    def _add_news_sentiment_features(self, features: pd.DataFrame):
        """Add news-based sentiment features using free sources"""
        # Using dummy random values for now since we don't have API access
        # In practice, you would use news API data here
        features['News_Sentiment'] = np.random.normal(0.5, 0.1, size=len(features))
        features['News_Volume'] = np.random.normal(100, 10, size=len(features))
    
    def _add_aggregated_sentiment(self, features: pd.DataFrame):
        """Calculate aggregated sentiment scores"""
        # Technical sentiment aggregate
        tech_columns = ['RSI_Sentiment', 'MACD_Sentiment', 'Volume_Sentiment']
        features['Technical_Sentiment_Aggregate'] = features[tech_columns].mean(axis=1)
        
        # Market sentiment aggregate
        market_columns = ['Fear_Greed_Index', 'Put_Call_Proxy']
        features['Market_Sentiment_Aggregate'] = features[market_columns].mean(axis=1)
        
        # Overall sentiment score
        sentiment_columns = [
            'Technical_Sentiment_Aggregate',
            'Market_Sentiment_Aggregate',
            'News_Sentiment'
        ]
        features['Overall_Sentiment'] = features[sentiment_columns].mean(axis=1)
    
    @staticmethod
    def _scale_to_100(series: pd.Series) -> pd.Series:
        """Scale a series to range 0-100"""
        min_val = series.rolling(window=252, min_periods=1).min()
        max_val = series.rolling(window=252, min_periods=1).max()
        return ((series - min_val) / (max_val - min_val)) * 100

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    from data_loader import DataLoader
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Load market data
    loader = DataLoader(config)
    market_data = loader.get_market_data()
    
    # Calculate sentiment features
    sentiment_features = SentimentFeatures(config)
    features = sentiment_features.calculate_all_features(market_data)
    
    # Print feature info
    print("\nSentiment Feature Information:")
    print(features.info())
    
    # Print sample statistics
    print("\nSentiment Feature Statistics:")
    print(features.describe())