"""
Feature Generator Module

This module coordinates the generation of all features:
- Technical features
- Sentiment features
- Macro features
- Feature combination and selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from config import Config
from sklearn.feature_selection import SelectKBest, f_classif
from src.features.technical_features import TechnicalFeatures
from src.features.sentiment_features import SentimentFeatures
from src.features.macro_features import MacroFeatures

class FeatureGenerator:
    """Coordinates feature generation from all sources"""
    
    def __init__(self, config: Config):
        """
        Initialize FeatureGenerator with configuration
        
        Args:
            config: Configuration object containing all feature parameters
        """
        self.config = config
        self.setup_logging()
        
        # Initialize feature generators
        self.technical_generator = TechnicalFeatures(config)
        self.sentiment_generator = SentimentFeatures(config)
        self.macro_generator = MacroFeatures(config)
        
        # Store feature metadata
        self.feature_columns: Dict[str, List[str]] = {}
        self.selected_features: List[str] = []
    
    def setup_logging(self):
        """Configure logging for the feature generator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FeatureGenerator')
    
    def generate_all_features(self, 
                            market_data: pd.DataFrame,
                            macro_data: Optional[pd.DataFrame] = None,
                            sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate all features from available data sources
        
        Args:
            market_data: DataFrame with OHLCV data
            macro_data: Optional DataFrame with macro indicators
            sentiment_data: Optional DataFrame with sentiment data
            
        Returns:
            DataFrame with all generated features
        """
        try:
            # Generate features from each source
            technical_features = self.technical_generator.calculate_all_features(market_data)
            self.feature_columns['technical'] = technical_features.columns.tolist()
            
            features_df = technical_features.copy()
            
            # Add macro features if data is available
            if macro_data is not None and not macro_data.empty:
                macro_features = self.macro_generator.calculate_all_features(
                    macro_data, market_data
                )
                self.feature_columns['macro'] = macro_features.columns.tolist()
                features_df = pd.concat([features_df, macro_features], axis=1)
            
            # Add sentiment features if data is available
            if sentiment_data is not None and not sentiment_data.empty:
                sentiment_features = self.sentiment_generator.calculate_all_features(
                    market_data
                )
                self.feature_columns['sentiment'] = sentiment_features.columns.tolist()
                features_df = pd.concat([features_df, sentiment_features], axis=1)
            
            # Handle missing values in final dataset
            features_df = self._handle_missing_values(features_df)
            
            # Log feature generation summary
            self._log_feature_summary(features_df)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            raise
    
    def _handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the combined feature set"""
        # Calculate missing percentage for each feature
        missing_pct = features_df.isnull().mean()
        
        # Drop features with too many missing values (e.g., > 30%)
        columns_to_drop = missing_pct[missing_pct > 0.3].index
        if len(columns_to_drop) > 0:
            self.logger.warning(f"Dropping features with >30% missing values: {columns_to_drop.tolist()}")
            features_df = features_df.drop(columns=columns_to_drop)
        
        # Forward fill remaining missing values
        features_df = features_df.fillna(method='ffill')
        
        # Back fill any remaining missing values
        features_df = features_df.fillna(method='bfill')
        
        return features_df
    
    def select_features(self, 
                       features_df: pd.DataFrame, 
                       target: pd.Series,
                       n_features: Optional[int] = None) -> pd.DataFrame:
        """
        Select most important features using statistical tests
        
        Args:
            features_df: DataFrame with all features
            target: Series with target values
            n_features: Optional number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if n_features is None:
            n_features = len(features_df.columns) // 3  # Default to 1/3 of features
        
        try:
            # Remove any constant features
            constant_features = [col for col in features_df.columns 
                               if features_df[col].nunique() == 1]
            if constant_features:
                features_df = features_df.drop(columns=constant_features)
                self.logger.warning(f"Dropped constant features: {constant_features}")
            
            # Apply feature selection
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(features_df, target)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            self.selected_features = features_df.columns[selected_mask].tolist()
            
            # Log selection results
            self.logger.info(f"Selected {len(self.selected_features)} features")
            
            return features_df[self.selected_features]
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            raise
    
    def generate_feature_groups(self, 
                              features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split features into groups based on their type
        
        Args:
            features_df: DataFrame with all features
            
        Returns:
            Dictionary with feature groups
        """
        feature_groups = {}
        
        for group_name, columns in self.feature_columns.items():
            # Get available columns (some might have been dropped)
            available_columns = [col for col in columns if col in features_df.columns]
            if available_columns:
                feature_groups[group_name] = features_df[available_columns]
        
        return feature_groups
    
    def _log_feature_summary(self, features_df: pd.DataFrame):
        """Log summary of generated features"""
        total_features = len(features_df.columns)
        
        summary = [
            f"Total features generated: {total_features}",
            "\nFeatures by group:"
        ]
        
        for group, columns in self.feature_columns.items():
            available_columns = [col for col in columns if col in features_df.columns]
            summary.append(f"- {group}: {len(available_columns)} features")
        
        self.logger.info("\n".join(summary))

# Example usage
if __name__ == "__main__":
    from config import Config, load_validated_config
    from src.data.data_loader import DataLoader
    
    # Load configuration
    config = load_validated_config('config/parameters.yaml')
    
    # Load data
    loader = DataLoader(config)
    market_data = loader.get_market_data()
    macro_data = loader.get_macro_data()
    
    # Initialize feature generator
    feature_gen = FeatureGenerator(config)
    
    # Generate all features
    features = feature_gen.generate_all_features(
        market_data=market_data,
        macro_data=macro_data
    )
    
    # Create target variable (example: next day return direction)
    target = np.sign(market_data['Close'].pct_change().shift(-1))
    
    # Select best features
    selected_features = feature_gen.select_features(
        features,
        target.loc[features.index]
    )
    
    # Get feature groups
    feature_groups = feature_gen.generate_feature_groups(selected_features)
    
    # Print summary
    print("\nFeature Generation Summary:")
    print(f"Total features: {len(features.columns)}")
    print(f"Selected features: {len(selected_features.columns)}")
    print("\nFeature groups:")
    for group, df in feature_groups.items():
        print(f"{group}: {len(df.columns)} features")