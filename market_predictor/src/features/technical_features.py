"""
Technical Features Module

This module handles the calculation of all technical indicators including:
- Price-based features (Returns, Moving Averages)
- Volume-based features (OBV, Volume Ratios)
- Momentum indicators (RSI, MFI)
- Volatility indicators (ATR, Daily Volatility)
- Pattern recognition features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from config import Config

class TechnicalFeatures:
    """Handles calculation of all technical indicators and features"""
    
    def __init__(self, config: Config):
        """
        Initialize TechnicalFeatures with configuration
        
        Args:
            config: Configuration object containing feature parameters
        """
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the feature calculator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TechnicalFeatures')
    
    def calculate_all_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all configured technical features
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical features
        """
        features = pd.DataFrame(index=market_data.index)
        
        try:
            # Calculate return-based features
            self._add_return_features(market_data, features)
            
            # Calculate moving average features
            self._add_ma_features(market_data, features)
            
            # Calculate volume-based features
            self._add_volume_features(market_data, features)
            
            # Calculate volatility features
            self._add_volatility_features(market_data, features)
            
            # Calculate momentum features
            self._add_momentum_features(market_data, features)
            
            # Calculate pattern features
            self._add_pattern_features(market_data, features)
            
            # Calculate trend features
            self._add_trend_features(market_data, features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            raise
    
    def _add_return_features(self, market_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate return-based features"""
        # Simple returns
        features['Returns'] = market_data['Close'].pct_change()
        
        # Log returns
        features['Log_Returns'] = np.log(market_data['Close']/market_data['Close'].shift(1))
    
    def _add_ma_features(self, market_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate moving average features"""
        for window in self.config.technical.ma_windows:
            # Simple moving average
            ma = market_data['Close'].rolling(window=window).mean()
            features[f'MA_{window}'] = ma
            
            # MA slope (momentum)
            features[f'MA_{window}_Slope'] = ma - ma.shift(1)
            
            # Price distance from MA
            features[f'Price_Distance_{window}'] = (
                market_data['Close'] - ma
            ) / ma
    
    def _add_volume_features(self, market_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate volume-based features"""
        # On-Balance Volume (OBV)
        obv = (market_data['Volume'] * 
               np.where(market_data['Close'] > market_data['Close'].shift(1), 1, -1)
              ).cumsum()
        features['OBV'] = obv
        
        # Volume Ratio
        features['Volume_Ratio'] = (
            market_data['Volume'] / 
            market_data['Volume'].rolling(window=20).mean()
        )
        
        # Force Index with 13-period EMA
        force_index = market_data['Close'].diff() * market_data['Volume']
        features['Force_Index_EMA13'] = force_index.ewm(span=13).mean()
        
        # Volume Trend
        features['Vol_Trend'] = market_data['Volume'].pct_change()
        
        # Price-Volume Trend
        features['PV_Trend'] = features['Returns'] * features['Vol_Trend']
        
        # Price-Volume Divergence
        price_ma = market_data['Close'].rolling(window=20).mean()
        volume_ma = market_data['Volume'].rolling(window=20).mean()
        features['PV_Divergence'] = (
            (market_data['Close'] - price_ma) / price_ma -
            (market_data['Volume'] - volume_ma) / volume_ma
        )
    
    def _add_volatility_features(self, market_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate volatility features"""
        # Daily Volatility
        features['Daily_Volatility'] = features['Returns'].rolling(window=20).std()
        
        # High-Low Range
        features['High_Low_Range'] = (
            market_data['High'] - market_data['Low']
        ) / market_data['Close']
        
        # Daily Gap
        features['Daily_Gap'] = (
            market_data['Open'] - market_data['Close'].shift(1)
        ) / market_data['Close'].shift(1)
        
        # ATR calculation
        tr1 = market_data['High'] - market_data['Low']
        tr2 = abs(market_data['High'] - market_data['Close'].shift(1))
        tr3 = abs(market_data['Low'] - market_data['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['ATR'] = tr.rolling(window=14).mean()
    
    def _add_momentum_features(self, market_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate momentum features"""
        # RSI for different periods
        for window in self.config.technical.rsi_windows:
            delta = market_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            features[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        # Money Flow Index (MFI)
        typical_price = (
            market_data['High'] + market_data['Low'] + market_data['Close']
        ) / 3
        money_flow = typical_price * market_data['Volume']
        
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(14).sum()
        negative_flow = money_flow.where(delta < 0, 0).rolling(14).sum()
        
        mfi_ratio = positive_flow / negative_flow
        features['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    def _add_pattern_features(self, market_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate pattern recognition features"""
        # Higher Highs
        features['Higher_High'] = (
            market_data['High'] > market_data['High'].shift(1)
        ).astype(int)
        
        # Lower Lows
        features['Lower_Low'] = (
            market_data['Low'] < market_data['Low'].shift(1)
        ).astype(int)
        
        # Inside Day
        features['Inside_Day'] = (
            (market_data['High'] <= market_data['High'].shift(1)) & 
            (market_data['Low'] >= market_data['Low'].shift(1))
        ).astype(int)
    
    def _add_trend_features(self, market_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate trend-related features"""
        # Bullish/Bearish Momentum
        close_ma20 = market_data['Close'].rolling(window=20).mean()
        features['Bullish_Momentum'] = (
            market_data['Close'] > close_ma20
        ).astype(int)
        features['Bearish_Momentum'] = (
            market_data['Close'] < close_ma20
        ).astype(int)
        
        # Trend Strength
        features['Trend_Strength'] = abs(
            features['Returns'].rolling(window=20).mean() / 
            features['Returns'].rolling(window=20).std()
        )

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    from data_loader import DataLoader
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Load market data
    loader = DataLoader(config)
    market_data = loader.get_market_data()
    
    # Calculate features
    tech_features = TechnicalFeatures(config)
    features = tech_features.calculate_all_features(market_data)
    
    # Print feature info
    print("\nFeature Information:")
    print(features.info())
    
    # Print sample statistics
    print("\nFeature Statistics:")
    print(features.describe())