"""
Logger Module

This module provides consistent logging functionality across the project:
- Custom formatting
- Multiple handlers (file and console)
- Different logging levels per module
- Log rotation
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import os
import sys

class ProjectLogger:
    """Custom logger for the Market Predictor project"""
    
    # Default log format
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Log levels mapping
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self,
                 name: str,
                 level: str = 'INFO',
                 log_file: Optional[str] = None,
                 console_output: bool = True,
                 format_string: Optional[str] = None):
        """
        Initialize logger
        
        Args:
            name: Logger name (usually module name)
            level: Logging level
            log_file: Optional file path for logging
            console_output: Whether to output to console
            format_string: Optional custom format string
        """
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LOG_LEVELS.get(level.upper(), logging.INFO))
        
        # Use custom format or default
        self.formatter = logging.Formatter(
            format_string or self.DEFAULT_FORMAT
        )
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Add console handler if requested
        if console_output:
            self._add_console_handler()
        
        # Add file handler if log file specified
        if log_file:
            self._add_file_handler(log_file)
    
    def _add_console_handler(self):
        """Add handler for console output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self, log_file: str):
        """Add handler for file output with rotation"""
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        return self.logger

def setup_project_logger(module_name: str,
                        log_dir: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to setup a logger with project defaults
    
    Args:
        module_name: Name of the module
        log_dir: Optional directory for log files
        
    Returns:
        Configured logger instance
    """
    # Default log directory is 'logs' in project root
    if log_dir is None:
        log_dir = 'logs'
    
    # Create log filename based on module and date
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = Path(log_dir) / f"{module_name}_{date_str}.log"
    
    # Create and return logger
    project_logger = ProjectLogger(
        name=module_name,
        level='INFO',
        log_file=str(log_file),
        console_output=True
    )
    
    return project_logger.get_logger()

class LoggerManager:
    """Manager class for handling multiple loggers"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls,
                  name: str,
                  level: str = 'INFO',
                  log_dir: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger
        
        Args:
            name: Logger name
            level: Logging level
            log_dir: Optional directory for log files
            
        Returns:
            Logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = setup_project_logger(name, log_dir)
            cls._loggers[name].setLevel(ProjectLogger.LOG_LEVELS.get(level.upper()))
        
        return cls._loggers[name]

# Example usage
if __name__ == "__main__":
    # Example 1: Basic logger setup
    logger = setup_project_logger('example_module')
    logger.info("This is an info message")
    logger.error("This is an error message")
    
    # Example 2: Custom logger with specific configuration
    custom_logger = ProjectLogger(
        name='custom_module',
        level='DEBUG',
        log_file='logs/custom.log',
        format_string='%(asctime)s - %(levelname)s - %(message)s'
    ).get_logger()
    
    custom_logger.debug("Debug message")
    custom_logger.info("Info message")
    
    # Example 3: Using LoggerManager
    manager_logger = LoggerManager.get_logger('managed_module', 'INFO', 'logs')
    manager_logger.info("Message from managed logger")
    
    # Example log messages
    logger.info("Starting data processing")
    logger.debug("Processing step 1")
    logger.warning("Resource usage high")
    logger.error("Failed to process data", exc_info=True)
    
    # Example with formatting
    data = {"symbol": "AAPL", "price": 150.25}
    logger.info(f"Processing trade: {data}")