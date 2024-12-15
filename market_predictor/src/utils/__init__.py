"""
Utils Module

This module provides utility functions and classes for:
- Logging configuration
- Performance metrics
- Constants and configurations
- Helper functions
"""

import pandas as pd
from .constants import (
    TimeFrame,
    TECHNICAL_FEATURES,
    MACRO_INDICATORS,
    SENTIMENT_SOURCES,
    MODEL_PARAMS,
    METRICS,
    CACHE_CONFIG,
    RATE_LIMITS
)

from .logger import (
    setup_project_logger,
    ProjectLogger,
    LoggerManager
)

from .metrics import (
    ModelMetrics,
    TradingMetrics,
    RiskMetrics,
    DataQualityMetrics
)

__all__ = [
    # Constants
    'TimeFrame',
    'TECHNICAL_FEATURES',
    'MACRO_INDICATORS',
    'SENTIMENT_SOURCES',
    'MODEL_PARAMS',
    'METRICS',
    
    # Logging
    'setup_project_logger',
    'ProjectLogger',
    'LoggerManager',
    
    # Metrics
    'ModelMetrics',
    'TradingMetrics',
    'RiskMetrics',
    'DataQualityMetrics',
    
    # Helper functions
    'validate_data',
    'get_project_path',
    'setup_environment'
]

import os
from pathlib import Path
from typing import Union, Dict, Any

def validate_data(data: Dict[str, Any]) -> bool:
    """
    Validate data structure and contents
    
    Args:
        data: Dictionary of data to validate
        
    Returns:
        Boolean indicating if data is valid
    """
    required_fields = ['open', 'high', 'low', 'close', 'volume']
    
    try:
        # Check for required fields
        if not all(field in data for field in required_fields):
            return False
            
        # Check for non-empty data
        if any(data[field].empty for field in required_fields):
            return False
            
        # Check for valid date index
        if not isinstance(data[required_fields[0]].index, pd.DatetimeIndex):
            return False
            
        return True
        
    except Exception:
        return False

def get_project_path() -> Path:
    """
    Get the absolute path to the project root directory
    
    Returns:
        Path object pointing to project root
    """
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent

def setup_environment(config: dict) -> None:
    """
    Setup project environment including directories and logging
    
    Args:
        config: Configuration dictionary
    """
    project_path = get_project_path()
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed',
        'data/cache',
        'logs',
        'models',
        'reports'
    ]
    
    for directory in directories:
        dir_path = project_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = project_path / 'logs' / 'main.log'
    logger = setup_project_logger('main', str(log_path))
    logger.info('Environment setup completed')

# Version of the utils module
__version__ = '0.1.0'

# Module configurations
UTILS_CONFIG = {
    'log_rotation': True,
    'log_retention': 30,  # days
    'cache_enabled': True,
    'default_timeframe': TimeFrame.DAILY
}