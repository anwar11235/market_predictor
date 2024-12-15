"""
Metrics Module

This module provides functionality for:
- Model evaluation metrics
- Trading performance metrics
- Risk metrics
- Data quality metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
)
from scipy import stats

class ModelMetrics:
    """Class for calculating model performance metrics"""
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Add probability metrics if available
        if y_prob is not None:
            metrics['log_loss'] = log_loss(y_true, y_prob)
            
            # ROC AUC for binary classification
            if y_prob.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        
        return metrics

    @staticmethod
    def regression_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict:
        """
        Calculate regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics

class TradingMetrics:
    """Class for calculating trading performance metrics"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate simple returns"""
        return prices.pct_change()
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns"""
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series,
                             risk_free_rate: float = 0.0,
                             periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
            
        Returns:
            Sharpe ratio
        """
        rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = returns - rf_per_period
        return np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
            
        Returns:
            Sortino ratio
        """
        rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = returns - rf_per_period
        downside = excess_returns[excess_returns < 0].std()
        return np.sqrt(periods_per_year) * (excess_returns.mean() / downside)
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            prices: Series of prices
            
        Returns:
            Maximum drawdown percentage
        """
        rolling_max = prices.expanding().max()
        drawdowns = prices / rolling_max - 1.0
        return drawdowns.min()
    
    @staticmethod
    def calculate_win_rate(trades: pd.Series) -> float:
        """
        Calculate win rate
        
        Args:
            trades: Series of trade returns
            
        Returns:
            Win rate percentage
        """
        winning_trades = (trades > 0).sum()
        total_trades = len(trades)
        return winning_trades / total_trades if total_trades > 0 else 0.0

class RiskMetrics:
    """Class for calculating risk metrics"""
    
    @staticmethod
    def calculate_var(returns: pd.Series,
                     confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level for VaR
            
        Returns:
            VaR value
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series,
                      confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level for CVaR
            
        Returns:
            CVaR value
        """
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_beta(returns: pd.Series,
                      market_returns: pd.Series) -> float:
        """
        Calculate beta relative to market
        
        Args:
            returns: Series of returns
            market_returns: Series of market returns
            
        Returns:
            Beta value
        """
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance

class DataQualityMetrics:
    """Class for calculating data quality metrics"""
    
    @staticmethod
    def calculate_missing_percentages(df: pd.DataFrame) -> pd.Series:
        """
        Calculate percentage of missing values per column
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Series with missing percentages
        """
        return (df.isnull().sum() / len(df)) * 100
    
    @staticmethod
    def detect_outliers(series: pd.Series,
                       n_std: float = 3.0) -> pd.Series:
        """
        Detect outliers using standard deviation method
        
        Args:
            series: Series to analyze
            n_std: Number of standard deviations for threshold
            
        Returns:
            Boolean series indicating outliers
        """
        mean = series.mean()
        std = series.std()
        return (series - mean).abs() > (n_std * std)
    
    @staticmethod
    def calculate_data_staleness(df: pd.DataFrame,
                               timestamp_col: str) -> pd.Timedelta:
        """
        Calculate data staleness
        
        Args:
            df: DataFrame to analyze
            timestamp_col: Name of timestamp column
            
        Returns:
            Time since last update
        """
        latest_timestamp = pd.to_datetime(df[timestamp_col]).max()
        return pd.Timestamp.now() - latest_timestamp

# Example usage
if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    prices = 100 * (1 + returns).cumprod()
    
    # Calculate trading metrics
    trading_metrics = {
        'sharpe': TradingMetrics.calculate_sharpe_ratio(returns),
        'sortino': TradingMetrics.calculate_sortino_ratio(returns),
        'max_drawdown': TradingMetrics.calculate_max_drawdown(prices)
    }
    
    print("\nTrading Metrics:")
    for metric, value in trading_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Calculate risk metrics
    risk_metrics = {
        'var_95': RiskMetrics.calculate_var(returns),
        'cvar_95': RiskMetrics.calculate_cvar(returns)
    }
    
    print("\nRisk Metrics:")
    for metric, value in risk_metrics.items():
        print(f"{metric}: {value:.4f}")