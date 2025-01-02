"""
Colab Setup Module

This module handles Google Colab specific setup and configurations.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import logging

def setup_colab_environment(
    project_root: Optional[str] = None,
    mount_drive: bool = True,
    install_dependencies: bool = True
) -> Path:
    """
    Set up the Google Colab environment for the market predictor project.
    
    Args:
        project_root: Optional path to project root. If None, will create in /content
        mount_drive: Whether to mount Google Drive
        install_dependencies: Whether to install required packages
        
    Returns:
        Path to project root
    """
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False
        return Path(project_root) if project_root else Path.cwd()
    
    if not is_colab:
        return Path(project_root) if project_root else Path.cwd()
    
    # Mount Google Drive if requested
    if mount_drive:
        from google.colab import drive
        drive.mount('/content/drive')
    
    # Set up project directory
    if project_root is None:
        project_root = '/content/market_predictor'
    
    project_path = Path(project_root)
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Install dependencies if requested
    if install_dependencies:
        os.system('pip install -q yfinance pandas numpy scikit-learn xgboost lightgbm fredapi')
    
    # Add project root to Python path
    if str(project_path) not in sys.path:
        sys.path.append(str(project_path))
    
    return project_path

def setup_data_directories(
    project_root: Path,
    use_drive: bool = True,
    drive_root: str = '/content/drive/MyDrive/market_predictor_data'
) -> dict:
    """
    Set up data directories for the project.
    
    Args:
        project_root: Path to project root
        use_drive: Whether to use Google Drive for data storage
        drive_root: Root path in Google Drive for data storage
        
    Returns:
        Dictionary with data directory paths
    """
    if use_drive:
        data_root = Path(drive_root)
    else:
        data_root = project_root / 'data'
    
    # Create directory structure
    dirs = {
        'raw': data_root / 'raw',
        'processed': data_root / 'processed',
        'models': data_root / 'models',
        'cache': data_root / 'cache'
    }
    
    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def setup_logging(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Optional directory for log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('market_predictor')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_dir is provided
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'market_predictor.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 