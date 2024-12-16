"""
Test Suite Initialization for Market Predictor

This module organizes and exposes the test suite for:
- Data loading and processing
- Feature engineering
- Model implementation
"""

from .test_data_loader import *
from .test_features import *
from .test_models import *

# Version of the test suite
__version__ = '0.1.0'

# Test configurations
TEST_CONFIG = {
    'random_seed': 42,
    'test_data_size': 1000,
    'test_features': 20,
    'test_classes': 2
}

# Test data parameters
SAMPLE_DATA_CONFIG = {
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'frequency': 'daily'
}

# Define test groups for selective testing
TEST_GROUPS = {
    'quick': [
        'test_data_loader.py::test_data_loader_initialization',
        'test_features.py::test_feature_consistency',
        'test_models.py::test_model_creation'
    ],
    'full': [
        'test_data_loader.py',
        'test_features.py',
        'test_models.py'
    ],
    'data': [
        'test_data_loader.py'
    ],
    'features': [
        'test_features.py'
    ],
    'models': [
        'test_models.py'
    ]
}

def run_test_group(group: str):
    """
    Run a specific group of tests
    
    Args:
        group: Name of the test group to run
    """
    import pytest
    
    if group not in TEST_GROUPS:
        raise ValueError(f"Invalid test group. Available groups: {list(TEST_GROUPS.keys())}")
    
    pytest.main(TEST_GROUPS[group])

def run_all_tests():
    """Run all tests with coverage report"""
    import pytest
    pytest.main(['--cov=src', '--cov-report=term-missing'])

# Common test utilities
def create_sample_data(size: int = 1000):
    """Create sample data for testing"""
    import numpy as np
    import pandas as pd
    
    return pd.DataFrame({
        'feature_' + str(i): np.random.randn(size)
        for i in range(TEST_CONFIG['test_features'])
    })

def create_sample_target(size: int = 1000):
    """Create sample target for testing"""
    import numpy as np
    return np.random.randint(0, TEST_CONFIG['test_classes'], size=size)

# Test fixtures available to all tests
import pytest

@pytest.fixture(scope='session')
def global_config():
    """Global configuration fixture"""
    from config import load_validated_config
    return load_validated_config('config/parameters.yaml')

@pytest.fixture(scope='session')
def sample_data():
    """Sample data fixture"""
    return create_sample_data()

@pytest.fixture(scope='session')
def sample_target():
    """Sample target fixture"""
    return create_sample_target()