"""
Market Predictor Data Module

This module provides functionality for loading and processing market data,
including technical, fundamental, and sentiment data sources.
"""

from .data_loader import DataLoader
from .data_processor import DataProcessor

__all__ = [
    'DataLoader',
    'DataProcessor',
    'load_and_process_data'
]

# Version of the data module
__version__ = '0.1.0'

def load_and_process_data(config):
    """
    Convenience function to load and process all data in one go.
    
    Args:
        config: Configuration object containing all settings
        
    Returns:
        tuple: (train_data, val_data, test_data), each a dictionary containing
        processed and scaled DataFrames for different data types
    """
    # Initialize components
    loader = DataLoader(config)
    processor = DataProcessor(config)
    
    try:
        # Load raw data
        raw_data = loader.get_all_data()
        
        # Process the data
        processed_data = processor.process_all_data(raw_data)
        
        # Split into train, validation, and test sets
        train_data, val_data, test_data = processor.split_data(processed_data)
        
        return train_data, val_data, test_data
    
    except Exception as e:
        raise RuntimeError(f"Error in data loading and processing: {str(e)}")

# Optional: Add any module-level configuration or initialization here
DEFAULT_CACHE_DIR = "./data/cache"