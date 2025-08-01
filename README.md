# AI-Powered Product Trend Prediction for E-commerce

A comprehensive machine learning solution for predicting product trends in e-commerce businesses using multiple AI models and real-time data analysis.

## Features

- **Multi-Model Approach**: Combines LSTM, XGBoost, and LightGBM for robust predictions
- **Real-time Analytics**: Live trend monitoring and prediction updates
- **Interactive Dashboard**: Web-based visualization of trends and predictions
- **REST API**: Easy integration with existing e-commerce platforms
- **Feature Engineering**: Advanced feature extraction from sales, seasonal, and market data
- **Model Evaluation**: Comprehensive metrics and performance tracking

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Application**
```bash
python main.py
```

3. **Access Dashboard**
Open http://localhost:8000 in your browser

## API Endpoints

- `GET /`: Dashboard interface
- `POST /predict`: Get trend predictions for specific products
- `GET /trends/{product_id}`: Historical trend data
- `POST /retrain`: Retrain models with new data

## Model Architecture

The system uses an ensemble of three models:
1. **LSTM Neural Network**: Captures sequential patterns and seasonality
2. **XGBoost**: Handles complex feature interactions
3. **LightGBM**: Fast gradient boosting for real-time predictions

## Data Requirements

The model expects the following data structure:
- Product sales history
- Seasonal indicators
- Market trends
- External factors (holidays, promotions)
- Competitor data (optional)

## Usage Example

```python
from models.trend_predictor import TrendPredictor

predictor = TrendPredictor()
predictions = predictor.predict(product_id="PROD123", days_ahead=30)
print(f"Predicted trend: {predictions}")
```

## Project Structure

```
├── models/           # ML models and training scripts
├── data/            # Data processing and feature engineering
├── api/             # FastAPI endpoints
├── dashboard/       # Web interface
├── utils/           # Helper functions
└── tests/           # Unit tests
```

