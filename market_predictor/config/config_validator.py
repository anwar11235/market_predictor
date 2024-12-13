from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import yaml
import re

class ConfigValidator:
    """Validates the configuration YAML file"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_config_file(self, config_path: str) -> bool:
        """
        Main validation method for the config file.
        Returns True if validation passes, False otherwise.
        """
        try:
            # Check if file exists
            path = Path(config_path)
            if not path.exists():
                self.errors.append(f"Configuration file not found: {config_path}")
                return False
            
            # Load YAML
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate main sections exist
            required_sections = ['data', 'technical', 'macro', 'sentiment', 'model']
            for section in required_sections:
                if section not in config:
                    self.errors.append(f"Missing required section: {section}")
            
            if self.errors:
                return False
            
            # Validate each section
            self._validate_data_config(config['data'])
            self._validate_technical_config(config['technical'])
            self._validate_macro_config(config['macro'])
            self._validate_sentiment_config(config['sentiment'])
            self._validate_model_config(config['model'])
            
            return len(self.errors) == 0
            
        except yaml.YAMLError as e:
            self.errors.append(f"Error parsing YAML file: {str(e)}")
            return False
    
    def _validate_data_config(self, config: Dict[str, Any]) -> None:
        """Validate data configuration section"""
        required_fields = [
            'ticker', 'start_date', 'training_start', 
            'validation_start', 'test_start', 'data_frequency'
        ]
        
        # Check required fields
        self._check_required_fields(config, required_fields, 'data')
        
        # Validate dates
        date_fields = ['start_date', 'training_start', 'validation_start', 'test_start']
        for field in date_fields:
            if field in config and config[field]:
                if not self._is_valid_date(config[field]):
                    self.errors.append(f"Invalid date format in {field}: {config[field]}")
        
        # Validate chronological order
        if all(field in config for field in date_fields):
            dates = [config[field] for field in date_fields if config[field]]
            if not self._dates_in_order(dates):
                self.errors.append("Dates must be in chronological order")
        
        # Validate data frequency
        valid_frequencies = ['daily', 'weekly', 'monthly']
        if config.get('data_frequency') not in valid_frequencies:
            self.errors.append(f"Invalid data_frequency. Must be one of: {valid_frequencies}")
    
    def _validate_technical_config(self, config: Dict[str, Any]) -> None:
        """Validate technical analysis configuration section"""
        # Validate return types
        valid_return_types = ['Returns', 'Log_Returns']
        if 'return_types' in config:
            invalid_returns = [r for r in config['return_types'] if r not in valid_return_types]
            if invalid_returns:
                self.errors.append(f"Invalid return types: {invalid_returns}")
        
        # Validate windows
        window_fields = ['ma_windows', 'price_distance_windows', 'volatility_windows', 'rsi_windows']
        for field in window_fields:
            if field in config:
                if not all(isinstance(w, int) and w > 0 for w in config[field]):
                    self.errors.append(f"Invalid {field}: All windows must be positive integers")
        
        # Validate periods
        period_fields = ['atr_period', 'mfi_period']
        for field in period_fields:
            if field in config:
                if not isinstance(config[field], int) or config[field] <= 0:
                    self.errors.append(f"Invalid {field}: Must be a positive integer")
    
    def _validate_macro_config(self, config: Dict[str, Any]) -> None:
        """Validate macroeconomic configuration section"""
        # Validate FRED series
        if 'fred_series' in config:
            if not all(isinstance(s, str) for s in config['fred_series']):
                self.errors.append("All FRED series must be strings")
        
        # Validate update frequency
        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly']
        if config.get('macro_update_frequency') not in valid_frequencies:
            self.errors.append(f"Invalid macro_update_frequency. Must be one of: {valid_frequencies}")
        
        # Validate interpolation method
        valid_methods = ['forward_fill', 'backward_fill', 'linear', 'cubic']
        if config.get('interpolation_method') not in valid_methods:
            self.errors.append(f"Invalid interpolation_method. Must be one of: {valid_methods}")
    
    def _validate_sentiment_config(self, config: Dict[str, Any]) -> None:
        """Validate sentiment configuration section"""
        # Validate sentiment windows
        if 'sentiment_windows' in config:
            if not all(isinstance(w, int) and w > 0 for w in config['sentiment_windows']):
                self.errors.append("All sentiment windows must be positive integers")
        
        # Validate aggregation method
        valid_methods = ['simple_average', 'weighted_average', 'exponential']
        if config.get('sentiment_aggregation') not in valid_methods:
            self.errors.append(f"Invalid sentiment_aggregation. Must be one of: {valid_methods}")
    
    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        """Validate model configuration section"""
        # Validate basic numeric parameters
        numeric_params = {
            'target_horizon': ('int', 1, None),
            'cv_folds': ('int', 2, None),
            'batch_size': ('int', 1, None),
            'max_epochs': ('int', 1, None),
            'learning_rate': ('float', 0, 1),
            'validation_size': ('float', 0, 1)
        }
        
        for param, (param_type, min_val, max_val) in numeric_params.items():
            if param in config:
                value = config[param]
                if param_type == 'int' and not isinstance(value, int):
                    self.errors.append(f"{param} must be an integer")
                elif param_type == 'float' and not isinstance(value, float):
                    self.errors.append(f"{param} must be a float")
                
                if min_val is not None and value < min_val:
                    self.errors.append(f"{param} must be >= {min_val}")
                if max_val is not None and value > max_val:
                    self.errors.append(f"{param} must be <= {max_val}")
    
    def _check_required_fields(self, config: Dict[str, Any], fields: List[str], section: str) -> None:
        """Helper method to check required fields"""
        for field in fields:
            if field not in config:
                self.errors.append(f"Missing required field '{field}' in {section} section")
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Helper method to validate date format"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def _dates_in_order(self, dates: List[str]) -> bool:
        """Helper method to check if dates are in chronological order"""
        return sorted(dates) == dates
    
    def print_validation_results(self) -> None:
        """Print validation results"""
        if self.errors:
            print("\nValidation Errors:")
            for error in self.errors:
                print(f"❌ {error}")
        
        if self.warnings:
            print("\nValidation Warnings:")
            for warning in self.warnings:
                print(f"⚠️ {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ Configuration is valid!")

# Example usage
if __name__ == "__main__":
    validator = ConfigValidator()
    if validator.validate_config_file('config/parameters.yaml'):
        print("Configuration validation successful!")
    else:
        validator.print_validation_results()