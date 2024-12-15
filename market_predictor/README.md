# Market Predictor

A comprehensive market prediction system that combines multiple data sources, advanced feature engineering, and ensemble modeling to predict market movements. The system integrates technical analysis, sentiment analysis, and macroeconomic indicators to provide well-rounded market insights.

## Features

### Data Integration
- **Market Data**: Real-time and historical price data using various financial APIs
- **News Sentiment**: Integration with multiple news sources (NewsAPI, Alpha Vantage, Finnhub)
- **Social Media Sentiment**: Analysis from Reddit and Twitter
- **Macroeconomic Indicators**: Integration with FRED and other economic data sources

### Feature Engineering
- **Technical Indicators**: Comprehensive set of technical analysis features
- **Sentiment Analysis**: Advanced sentiment processing from multiple sources
- **Macroeconomic Features**: Economic indicators and their derivatives
- **Feature Selection**: Automated feature importance and selection

### Model Architecture
- **Ensemble Modeling**: Combination of multiple model types
- **Dynamic Weighting**: Adaptive model weight adjustment
- **Performance Tracking**: Comprehensive model performance metrics
- **Risk Management**: Integration of risk metrics and constraints

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/market_predictor.git
cd market_predictor
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure API keys
- Copy `config/parameters.yaml.example` to `config/parameters.yaml`
- Add your API keys and configurations

## Project Structure

```
market_predictor/
│
├── src/
│   ├── data/                 # Data loading and processing
│   ├── features/            # Feature engineering
│   ├── models/              # Model implementations
│   ├── integrations/        # External API integrations
│   └── utils/               # Utility functions
│
├── config/                  # Configuration files
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
└── requirements.txt        # Project dependencies
```

## Usage

### Basic Usage

```python
from market_predictor import create_prediction_pipeline

# Initialize pipeline
data_loader, feature_gen, model = create_prediction_pipeline(config)

# Get predictions
predictions = model.predict(data)
```

### Example Notebook
See `notebooks/01_data_collection.ipynb` for a complete example of data collection and model training.

## Configuration

The system can be configured using `config/parameters.yaml`. Key configuration sections:

- Data Sources
- Feature Engineering
- Model Parameters
- Trading Parameters

## API References

### Required API Keys
- NewsAPI
- Alpha Vantage
- Finnhub
- Twitter (optional)
- Reddit (optional)

### Setting Up API Keys
1. Obtain API keys from respective services
2. Add keys to `config/parameters.yaml`
3. Use `config.load_validated_config()` to load configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Testing

Run the test suite:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by various financial APIs
- Built with Python and several open-source libraries


---

**Note**: This project is for educational purposes only. Always perform your own due diligence before making investment decisions.