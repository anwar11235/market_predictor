"""
Data Processor Module

This module handles data preprocessing, cleaning, and preparation including:
- Data cleaning and validation
- Time series alignment
- Feature scaling and normalization
- Train/validation/test splitting
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
from datetime import datetime
from config import Config

class DataProcessor:
    """Main data processor class that handles all data preprocessing steps"""
    
    def __init__(self, config: Config):
        """
        Initialize DataProcessor with configuration
        
        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        self.setup_logging()
        self.scalers = {}
    
    def setup_logging(self):
        """Configure logging for the data processor"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataProcessor')

    def process_all_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all datasets
        
        Args:
            data: Dictionary of DataFrames containing market, macro, and sentiment data
            
        Returns:
            Dictionary of processed DataFrames
        """
        processed_data = {}
        
        # Process each dataset
        for data_type, df in data.items():
            if df is not None and not df.empty:
                processed_data[data_type] = self._process_dataset(df, data_type)
                self.logger.info(f"Processed {data_type} data")
        
        # Align all datasets to same date range
        processed_data = self._align_datasets(processed_data)
        
        return processed_data

    def _process_dataset(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Process individual dataset
        
        Args:
            df: DataFrame to process
            data_type: Type of data (market, macro, sentiment)
            
        Returns:
            Processed DataFrame
        """
        # Make copy to avoid modifying original
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Handle missing values based on data type
        df = self._handle_missing_values(df, data_type)
        
        # Remove outliers if specified
        df = self._handle_outliers(df, data_type)
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Handle missing values based on data type
        
        Args:
            df: DataFrame with missing values
            data_type: Type of data
            
        Returns:
            DataFrame with handled missing values
        """
        if data_type == 'market':
            # For market data, forward fill limited to 5 days
            df = df.fillna(method='ffill', limit=5)
            # If still missing, backward fill
            df = df.fillna(method='bfill', limit=5)
        
        elif data_type == 'macro':
            # For macro data, use specified interpolation method
            method = self.config.macro.interpolation_method
            if method == 'forward_fill':
                df = df.fillna(method='ffill')
            elif method == 'linear':
                df = df.interpolate(method='linear')
            
        elif data_type == 'sentiment':
            # For sentiment, forward fill limited to 3 days
            df = df.fillna(method='ffill', limit=3)
        
        # Drop any remaining rows with missing values
        df = df.dropna()
        
        return df

    def _handle_outliers(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Handle outliers using IQR method
        
        Args:
            df: DataFrame potentially containing outliers
            data_type: Type of data
            
        Returns:
            DataFrame with handled outliers
        """
        if data_type == 'market':
            # For market data, we keep all values as they're real prices
            return df
        
        # For other data types, handle outliers
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap outliers instead of removing them
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df

    def _align_datasets(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all datasets to the same date range
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Dictionary of aligned DataFrames
        """
        # Find common date range
        common_dates = None
        for df in data.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        # Align all datasets to common dates
        for key in data:
            data[key] = data[key].loc[sorted(common_dates)]
        
        return data

    def split_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], ...]:
        """
        Split data into train, validation, and test sets
        
        Args:
            data: Dictionary of processed DataFrames
            
        Returns:
            Tuple of dictionaries containing train, validation, and test data
        """
        train_data = {}
        val_data = {}
        test_data = {}
        
        for data_type, df in data.items():
            # Define split dates
            train_mask = df.index < pd.Timestamp(self.config.data.validation_start)
            val_mask = (df.index >= pd.Timestamp(self.config.data.validation_start)) & \
                      (df.index < pd.Timestamp(self.config.data.test_start))
            test_mask = df.index >= pd.Timestamp(self.config.data.test_start)
            
            # Split data
            train_data[data_type] = df[train_mask].copy()
            val_data[data_type] = df[val_mask].copy()
            test_data[data_type] = df[test_mask].copy()
            
            # Fit scaler on training data and transform all sets
            self.scalers[data_type] = RobustScaler()
            
            train_data[data_type] = pd.DataFrame(
                self.scalers[data_type].fit_transform(train_data[data_type]),
                index=train_data[data_type].index,
                columns=train_data[data_type].columns
            )
            
            val_data[data_type] = pd.DataFrame(
                self.scalers[data_type].transform(val_data[data_type]),
                index=val_data[data_type].index,
                columns=val_data[data_type].columns
            )
            
            test_data[data_type] = pd.DataFrame(
                self.scalers[data_type].transform(test_data[data_type]),
                index=test_data[data_type].index,
                columns=test_data[data_type].columns
            )
        
        return train_data, val_data, test_data

    def inverse_transform(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale
        
        Args:
            data: Scaled DataFrame
            data_type: Type of data
            
        Returns:
            DataFrame in original scale
        """
        if data_type not in self.scalers:
            raise ValueError(f"No scaler found for {data_type} data")
        
        return pd.DataFrame(
            self.scalers[data_type].inverse_transform(data),
            index=data.index,
            columns=data.columns
        )

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    from data_loader import DataLoader
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Initialize data loader and processor
    loader = DataLoader(config)
    processor = DataProcessor(config)
    
    # Load and process data
    raw_data = loader.get_all_data()
    processed_data = processor.process_all_data(raw_data)
    
    # Split data
    train_data, val_data, test_data = processor.split_data(processed_data)
    
    # Print shapes
    for data_type in train_data:
        print(f"\n{data_type.upper()} DATA SHAPES:")
        print(f"Train: {train_data[data_type].shape}")
        print(f"Validation: {val_data[data_type].shape}")
        print(f"Test: {test_data[data_type].shape}")