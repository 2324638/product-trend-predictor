# AI-Powered Product Trend Prediction for E-commerce

An advanced machine learning system for predicting product trends using ensemble models including **Prophet**, LSTM, XGBoost, and LightGBM. Built specifically for the Superstore dataset with real-time forecasting capabilities.

## 🚀 Features

### 🤖 **Multi-Model AI Ensemble**
- **Prophet**: Facebook's time series forecasting with seasonal decomposition
- **LSTM**: Deep learning for sequence modeling
- **XGBoost**: Gradient boosting for structured data
- **LightGBM**: Fast gradient boosting framework

### 📊 **Advanced Analytics**
- Real-time trend prediction with confidence intervals
- Seasonal trend analysis and holiday effects modeling
- Product-specific forecasting
- Interactive web dashboard with visualizations
- REST API for integration

### 🔧 **Data Processing**
- Superstore dataset integration
- Advanced feature engineering
- Automated data preprocessing
- Missing value handling
- Temporal feature creation

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd product-trend-predictor

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Manual Installation
```bash
# Core dependencies
pip install pandas numpy scikit-learn

# Machine Learning models
pip install xgboost lightgbm prophet

# Deep Learning (optional)
pip install tensorflow

# Web framework
pip install fastapi uvicorn

# Visualization
pip install matplotlib seaborn plotly
```

## 🎯 Usage

### 1. **Start the Application**
```bash
python main.py
```

### 2. **Access the Dashboard**
- Open http://localhost:8000 in your browser
- Navigate to "Data Management" tab
- Click "Load Superstore Data" to load your dataset
- Click "Train Model" to train the AI models
- Go to "Predictions" tab to make forecasts

### 3. **API Access**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 🔮 Prophet Integration

### **Prophet Features**
- **Time Series Forecasting**: Predicts future values based on historical patterns
- **Seasonal Decomposition**: Separates trend, seasonality, and noise
- **Holiday Effects**: Models holiday impacts on sales
- **Confidence Intervals**: Provides uncertainty estimates
- **Multiple Seasonalities**: Daily, weekly, and yearly patterns

### **Prophet Usage Example**
```python
from data.superstore_loader import SuperstoreDataLoader
from models.prophet_predictor import ProphetPredictor

# Load Superstore data
loader = SuperstoreDataLoader()
df = loader.get_sample_data(n_products=10, days=365)

# Initialize and train Prophet
prophet = ProphetPredictor()
prophet.fit(df, fit_global=True, fit_products=True)

# Make predictions
predictions = prophet.predict(df, periods=30)

# Access results
global_forecast = predictions['global']
product_forecasts = predictions['products']
```

### **Prophet Configuration**
```python
prophet = ProphetPredictor(
    daily_seasonality=True,      # Daily patterns
    weekly_seasonality=True,     # Weekly patterns  
    yearly_seasonality=True,     # Yearly patterns
    changepoint_prior_scale=0.05, # Trend flexibility
    seasonality_prior_scale=10.0  # Seasonality flexibility
)
```

## 📊 Superstore Dataset

The system is optimized for the Superstore dataset with:
- **9,800+ records** of sales data
- **1,610 unique products** across 3 categories
- **4-year time span** (2015-2018)
- **Multiple dimensions**: Product, Category, Region, Customer Segment

### **Data Preprocessing**
- Automatic quantity estimation from sales data
- Profit margin calculation by category
- Temporal feature engineering
- Market simulation for external factors
- Holiday and seasonal feature creation

## 🏗️ Architecture

### **Data Flow**
```
Superstore CSV → Data Loader → Preprocessor → Models → Predictions
```

### **Model Ensemble**
- **Prophet**: 25% weight (time series forecasting)
- **XGBoost**: 25% weight (structured data)
- **LightGBM**: 25% weight (gradient boosting)
- **LSTM**: 25% weight (sequence modeling)

### **API Endpoints**
- `GET /`: Dashboard interface
- `GET /health`: System health check
- `POST /predict`: Make predictions
- `GET /trends/{product_id}`: Product-specific trends
- `POST /train`: Train models
- `GET /generate-sample-data`: Load Superstore data

## 🔧 Configuration

### **Model Parameters**
```python
# Trend Predictor
predictor = TrendPredictor(sequence_length=30)

# Prophet
prophet = ProphetPredictor(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

# Data Loader
loader = SuperstoreDataLoader()
df = loader.get_sample_data(n_products=20, days=365)
```

### **Environment Variables**
```bash
# Optional: Set model save path
MODEL_SAVE_PATH=models/saved

# Optional: Set data path
SUPERSTORE_DATA_PATH=data/superstore.csv
```

## 📈 Performance

### **Model Accuracy** (example results)
- **XGBoost**: MAE=0.280, RMSE=0.511, R²=0.621
- **LightGBM**: MAE=4.583, RMSE=4.658, R²=-30.556
- **Prophet**: Time series forecasting with confidence intervals
- **Ensemble**: Combined prediction with weighted averaging

### **Forecasting Capabilities**
- **Short-term**: 7-30 days ahead
- **Medium-term**: 1-3 months ahead
- **Long-term**: 3-12 months ahead
- **Confidence intervals**: 95% prediction bands

## 🛠️ Development

### **Project Structure**
```
├── api/                 # FastAPI application
├── data/               # Data processing modules
│   ├── superstore_loader.py
│   ├── preprocessor.py
│   └── models.py
├── models/             # ML models
│   ├── trend_predictor.py
│   ├── prophet_predictor.py
│   └── saved/          # Trained models
├── dashboard/          # Web interface
├── utils/              # Utility functions
└── requirements.txt    # Dependencies
```

### **Adding New Models**
1. Create model class in `models/`
2. Implement `fit()` and `predict()` methods
3. Add to ensemble in `TrendPredictor`
4. Update weights and evaluation

### **Extending Data Sources**
1. Create new data loader in `data/`
2. Implement preprocessing pipeline
3. Update API endpoints
4. Test with dashboard

## 🚀 Deployment

### **Docker Deployment**
```bash
# Build image
docker build -t trend-predictor .

# Run container
docker run -p 8000:8000 trend-predictor
```

### **Production Considerations**
- Use production WSGI server (Gunicorn)
- Set up monitoring and logging
- Configure database for model persistence
- Implement caching for predictions
- Set up CI/CD pipeline

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes and test
4. Submit pull request

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example notebooks

---

**Built with ❤️ for e-commerce trend prediction using cutting-edge AI technologies.**

