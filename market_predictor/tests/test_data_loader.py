"""
Test suite for data loading and processing functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.data_loader import DataLoader
from src.utils import setup_project_logger
from config import Config, load_validated_config

@pytest.fixture
def config():
    """Fixture for test configuration"""
    return load_validated_config('config/parameters.yaml')

@pytest.fixture
def data_loader(config):
    """Fixture for data loader instance"""
    return DataLoader(config)

@pytest.fixture
def sample_market_data():
    """Fixture for sample market data"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    return pd.DataFrame({
        'Open': np.random.randn(len(dates)) * 10 + 100,
        'High': np.random.randn(len(dates)) * 10 + 101,
        'Low': np.random.randn(len(dates)) * 10 + 99,
        'Close': np.random.randn(len(dates)) * 10 + 100,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

def test_data_loader_initialization(data_loader):
    """Test DataLoader initialization"""
    assert isinstance(data_loader, DataLoader)
    assert hasattr(data_loader, 'config')
    assert hasattr(data_loader, 'logger')

def test_get_market_data(data_loader):
    """Test market data retrieval"""
    market_data = data_loader.get_market_data()
    
    # Check dataframe structure
    assert isinstance(market_data, pd.DataFrame)
    assert all(col in market_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    assert isinstance(market_data.index, pd.DatetimeIndex)
    
    # Check data integrity
    assert not market_data.empty
    assert market_data['High'].ge(market_data['Low']).all()
    assert market_data['Volume'].ge(0).all()

def test_get_macro_data(data_loader):
    """Test macroeconomic data retrieval"""
    macro_data = data_loader.get_macro_data()
    
    # Check basic structure
    assert isinstance(macro_data, pd.DataFrame)
    assert not macro_data.empty
    assert isinstance(macro_data.index, pd.DatetimeIndex)

def test_data_alignment(data_loader):
    """Test alignment of different data sources"""
    market_data = data_loader.get_market_data()
    macro_data = data_loader.get_macro_data()
    
    # Check date alignment
    assert market_data.index.dtype == macro_data.index.dtype
    assert market_data.index.freq == macro_data.index.freq

def test_missing_data_handling(data_loader, sample_market_data):
    """Test handling of missing data"""
    # Introduce missing values
    sample_data = sample_market_data.copy()
    sample_data.loc[sample_data.index[2], 'Close'] = np.nan
    
    # Test forward fill
    filled_data = data_loader._handle_missing_values(sample_data)
    assert not filled_data.isnull().any().any()
    assert filled_data.loc[sample_data.index[2], 'Close'] == \
           sample_data.loc[sample_data.index[1], 'Close']

def test_date_range_validation(data_loader):
    """Test date range validation"""
    market_data = data_loader.get_market_data()
    
    # Check date range
    assert market_data.index.min() >= pd.Timestamp(data_loader.config.data.start_date)
    if data_loader.config.data.end_date:
        assert market_data.index.max() <= pd.Timestamp(data_loader.config.data.end_date)

def test_cache_functionality(data_loader):
    """Test data caching functionality"""
    # First call to get data
    first_call = data_loader.get_market_data()
    
    # Second call should use cache
    second_call = data_loader.get_market_data()
    
    pd.testing.assert_frame_equal(first_call, second_call)

def test_data_frequency(data_loader):
    """Test data frequency consistency"""
    market_data = data_loader.get_market_data()
    
    # Check for daily data
    date_diffs = market_data.index[1:] - market_data.index[:-1]
    assert all(diff.days == 1 for diff in date_diffs)

def test_error_handling(data_loader):
    """Test error handling in data loading"""
    with pytest.raises(Exception):
        data_loader.get_market_data(ticker="INVALID_TICKER")

@pytest.mark.parametrize("ticker", ['AAPL', 'MSFT', 'GOOGL'])
def test_multiple_tickers(data_loader, ticker):
    """Test data loading for multiple tickers"""
    try:
        data = data_loader.get_market_data(ticker=ticker)
        assert not data.empty
    except Exception as e:
        pytest.fail(f"Failed to load data for {ticker}: {str(e)}")

if __name__ == '__main__':
    pytest.main([__file__])