"""
Macro Features Module

This module handles the calculation of macroeconomic features including:
- Economic indicators (GDP, CPI, etc.)
- Interest rate features
- Money supply metrics
- Employment data
- Market environment indicators
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from config import Config
from datetime import datetime, timedelta

class MacroFeatures:
    """Handles calculation and processing of macroeconomic features"""
    
    def __init__(self, config: Config):
        """
        Initialize MacroFeatures with configuration
        
        Args:
            config: Configuration object containing macro parameters
        """
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the macro analyzer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MacroFeatures')
    
    def calculate_all_features(self, 
                             macro_data: pd.DataFrame, 
                             market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all configured macro features
        
        Args:
            macro_data: DataFrame with raw macro indicators
            market_data: DataFrame with market data for alignment
            
        Returns:
            DataFrame with all macro features
        """
        features = pd.DataFrame(index=market_data.index)
        
        try:
            # Align macro data to market dates
            aligned_macro = self._align_macro_data(macro_data, market_data.index)
            
            # Calculate different types of macro features
            self._add_rate_features(aligned_macro, features)
            self._add_growth_features(aligned_macro, features)
            self._add_momentum_features(aligned_macro, features)
            self._add_relative_features(aligned_macro, features)
            self._add_composite_features(aligned_macro, features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating macro features: {e}")
            raise
    
    def _align_macro_data(self, macro_data: pd.DataFrame, 
                         market_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align macro data to market dates using specified interpolation method
        
        Args:
            macro_data: DataFrame with macro indicators
            market_dates: DatetimeIndex of market dates
            
        Returns:
            DataFrame with aligned macro data
        """
        aligned_data = pd.DataFrame(index=market_dates)
        
        for column in macro_data.columns:
            # Resample to daily frequency
            daily_data = macro_data[column].reindex(market_dates)
            
            # Apply specified interpolation method
            if self.config.macro.interpolation_method == 'forward_fill':
                aligned_data[column] = daily_data.fillna(method='ffill')
            elif self.config.macro.interpolation_method == 'linear':
                aligned_data[column] = daily_data.interpolate(method='linear')
            else:
                aligned_data[column] = daily_data.fillna(method='ffill')
        
        return aligned_data
    
    def _add_rate_features(self, macro_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate interest rate and rate-based features"""
        rate_indicators = ['DFF', 'T10Y2Y', 'T10YFF']
        
        for indicator in rate_indicators:
            if indicator in macro_data.columns:
                # Raw rates
                features[f'{indicator}_Rate'] = macro_data[indicator]
                
                # Rate changes
                features[f'{indicator}_Change'] = macro_data[indicator].diff()
                
                # Rate momentum (20-day change)
                features[f'{indicator}_Momentum'] = macro_data[indicator].diff(20)
                
                # Rate volatility
                features[f'{indicator}_Volatility'] = (
                    macro_data[indicator].rolling(window=20).std()
                )
    
    def _add_growth_features(self, macro_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate growth-related features"""
        growth_indicators = ['GDP', 'INDPRO', 'PCE']
        
        for indicator in growth_indicators:
            if indicator in macro_data.columns:
                # YoY Growth Rate
                features[f'{indicator}_YoY'] = (
                    macro_data[indicator].pct_change(periods=252)
                )
                
                # Growth momentum
                features[f'{indicator}_Momentum'] = (
                    features[f'{indicator}_YoY'].diff(20)
                )
                
                # Growth trend
                ma_slow = macro_data[indicator].rolling(window=252).mean()
                ma_fast = macro_data[indicator].rolling(window=63).mean()
                features[f'{indicator}_Trend'] = (ma_fast - ma_slow) / ma_slow
    
    def _add_momentum_features(self, macro_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate momentum-based features for macro indicators"""
        for column in macro_data.columns:
            # Short-term momentum (1-month)
            features[f'{column}_ST_Momentum'] = (
                macro_data[column].diff(21) / macro_data[column].shift(21)
            )
            
            # Medium-term momentum (3-month)
            features[f'{column}_MT_Momentum'] = (
                macro_data[column].diff(63) / macro_data[column].shift(63)
            )
            
            # Long-term momentum (6-month)
            features[f'{column}_LT_Momentum'] = (
                macro_data[column].diff(126) / macro_data[column].shift(126)
            )
    
    def _add_relative_features(self, macro_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate relative features between different indicators"""
        if all(x in macro_data.columns for x in ['CPI', 'PCE']):
            # Inflation spread
            features['Inflation_Spread'] = (
                macro_data['CPI'] - macro_data['PCE']
            )
        
        if all(x in macro_data.columns for x in ['GDP', 'M2']):
            # Money velocity proxy
            features['Money_Velocity'] = (
                macro_data['GDP'] / macro_data['M2']
            )
    
    def _add_composite_features(self, macro_data: pd.DataFrame, features: pd.DataFrame):
        """Calculate composite economic indicators"""
        # Economic Activity Index
        growth_columns = ['GDP', 'INDPRO', 'PCE']
        available_columns = [col for col in growth_columns if col in macro_data.columns]
        
        if available_columns:
            normalized_data = macro_data[available_columns].apply(
                lambda x: (x - x.rolling(252).mean()) / x.rolling(252).std()
            )
            features['Economic_Activity_Index'] = normalized_data.mean(axis=1)
        
        # Financial Conditions Index
        financial_columns = ['DFF', 'T10Y2Y', 'T10YFF']
        available_columns = [col for col in financial_columns if col in macro_data.columns]
        
        if available_columns:
            normalized_data = macro_data[available_columns].apply(
                lambda x: (x - x.rolling(252).mean()) / x.rolling(252).std()
            )
            features['Financial_Conditions_Index'] = normalized_data.mean(axis=1)

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    from data_loader import DataLoader
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Load data
    loader = DataLoader(config)
    market_data = loader.get_market_data()
    macro_data = loader.get_macro_data()
    
    # Calculate macro features
    macro_features = MacroFeatures(config)
    features = macro_features.calculate_all_features(macro_data, market_data)
    
    # Print feature info
    print("\nMacro Feature Information:")
    print(features.info())
    
    # Print sample statistics
    print("\nMacro Feature Statistics:")
    print(features.describe())