#!/bin/bash

# Create main project directory
mkdir -p market_predictor

# Create all directories
mkdir -p market_predictor/src/{data,features,models,utils}
mkdir -p market_predictor/config
mkdir -p market_predictor/notebooks
mkdir -p market_predictor/tests

# Create __init__.py files
touch market_predictor/src/__init__.py
touch market_predictor/src/data/__init__.py
touch market_predictor/src/features/__init__.py
touch market_predictor/src/models/__init__.py
touch market_predictor/src/utils/__init__.py
touch market_predictor/config/__init__.py
touch market_predictor/tests/__init__.py

# Create source files
touch market_predictor/src/data/{data_loader.py,data_processor.py}
touch market_predictor/src/features/{technical_features.py,sentiment_features.py,macro_features.py,feature_generator.py}
touch market_predictor/src/models/{base_model.py,model_factory.py,ensemble.py}
touch market_predictor/src/utils/{constants.py,logger.py,metrics.py}

# Create config files
touch market_predictor/config/{config.py,parameters.yaml}

# Create notebook files
touch market_predictor/notebooks/{01_data_collection.ipynb,02_feature_engineering.ipynb,03_model_development.ipynb,04_ensemble_training.ipynb,05_backtesting.ipynb}

# Create test files
touch market_predictor/tests/{test_data_loader.py,test_features.py,test_models.py}

# Create root level files
touch market_predictor/{requirements.txt,setup.py,README.md}

echo "Project structure created successfully!"
