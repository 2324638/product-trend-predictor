from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import io

# Local imports
from data.models import (
    PredictionRequest, PredictionResponse, ProductData, 
    SalesData, ModelMetrics, TrendAlert
)
from models.trend_predictor import TrendPredictor
from data.superstore_loader import SuperstoreDataLoader
from utils.database import DatabaseManager

# Initialize FastAPI app
app = FastAPI(
    title="AI Product Trend Prediction API",
    description="Advanced e-commerce product trend prediction using ensemble machine learning models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
trend_predictor = TrendPredictor()
db_manager = DatabaseManager()
current_dataset = None
model_trained = False

# Mount static files
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global current_dataset, model_trained
    
    # Try to load existing models
    try:
        trend_predictor.load_models()
        model_trained = True
        print("Loaded existing models")
    except (FileNotFoundError, Exception) as e:
        print(f"No existing models found: {e}")
        # Load complete Superstore dataset
        try:
            loader = SuperstoreDataLoader()
            current_dataset = loader.get_complete_dataset()
            print(f"Loaded complete Superstore dataset: {current_dataset.shape[0]} records for {current_dataset['product_id'].nunique()} products")
        except Exception as data_error:
            print(f"Could not load Superstore data: {data_error}")
            current_dataset = None


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard"""
    dashboard_path = "dashboard/index.html"
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>AI Trend Prediction Dashboard</title></head>
            <body>
                <h1>AI Product Trend Prediction Dashboard</h1>
                <p>Dashboard is loading... Please ensure dashboard files are available.</p>
                <p><a href="/docs">Access API Documentation</a></p>
            </body>
        </html>
        """)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_trained": model_trained,
        "timestamp": datetime.now().isoformat(),
        "dataset_available": current_dataset is not None
    }


@app.get("/metrics/quick")
async def get_quick_metrics():
    """Get basic metrics quickly for dashboard"""
    if current_dataset is None:
        return {
            "total_products": 0,
            "total_sales": 0,
            "total_revenue": 0.0,
            "avg_order_value": 0.0,
            "dataset_available": False
        }
    
    df = current_dataset
    
    # Calculate just the essential metrics quickly
    return {
        "total_products": int(df['product_id'].nunique()),
        "total_sales": int(df['quantity_sold'].sum()),
        "total_revenue": float(df['revenue'].sum()),
        "avg_order_value": float(df['revenue'].sum() / df['quantity_sold'].sum()),
        "dataset_available": True
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_product_trend(request: PredictionRequest):
    """Predict trends for a specific product"""
    if not model_trained:
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset available. Please upload data first.")
    
    try:
        prediction = trend_predictor.predict_product(
            product_id=request.product_id,
            df=current_dataset,
            days_ahead=request.days_ahead
        )
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/trends/{product_id}")
async def get_product_trends(product_id: str, days: int = 90):
    """Get historical trend data for a product"""
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset available")
    
    # Filter data for the product
    product_data = current_dataset[current_dataset['product_id'] == product_id].copy()
    
    if len(product_data) == 0:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    
    # Get recent data
    product_data = product_data.sort_values('date').tail(days)
    
    # Calculate basic statistics
    trends = {
        "product_id": product_id,
        "data_points": len(product_data),
        "date_range": {
            "start": product_data['date'].min().isoformat(),
            "end": product_data['date'].max().isoformat()
        },
        "statistics": {
            "avg_daily_sales": float(product_data['quantity_sold'].mean()),
            "total_sales": int(product_data['quantity_sold'].sum()),
            "avg_revenue": float(product_data['revenue'].mean()),
            "total_revenue": float(product_data['revenue'].sum()),
        },
        "historical_data": product_data[['date', 'quantity_sold', 'revenue']].to_dict('records')
    }
    
    return trends


@app.get("/products")
async def get_products():
    """Get list of available products from Superstore dataset"""
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset available")
    

    
    # Group by product_id and aggregate Superstore data
    products = current_dataset.groupby('product_id').agg({
        'product_name': 'first',  # Use product_name from Superstore
        'category': 'first',
        'sub_category': 'first',  # Add sub_category from Superstore
        'brand': 'first',
        'price': 'mean',  # Average price
        'quantity_sold': ['sum', 'mean'],
        'revenue': 'sum',
        'profit': 'sum',  # Add profit from Superstore
        'region': 'first',  # Add region from Superstore
        'state': 'first'   # Add state from Superstore
    }).round(2)
    
    # Flatten column names
    products.columns = [
        'product_name', 'category', 'sub_category', 'brand', 'avg_price',
        'total_sales', 'avg_daily_sales', 'total_revenue', 'total_profit',
        'region', 'state'
    ]
    products = products.reset_index()
    
    # Add calculated fields
    products['profit_margin'] = (products['total_profit'] / products['total_revenue'] * 100).round(2)
    products['avg_order_value'] = (products['total_revenue'] / products['total_sales']).round(2)
    
    # Sort by total revenue (most profitable first)
    products = products.sort_values('total_revenue', ascending=False)
    
    return {
        "total_products": len(products),
        "products": products.to_dict('records')
    }


@app.get("/analytics/overview")
async def get_analytics_overview():
    """Get overall analytics overview - optimized for dashboard"""
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset available")
    
    df = current_dataset
    
    # Calculate key metrics
    total_products = df['product_id'].nunique()
    total_sales = int(df['quantity_sold'].sum())
    total_revenue = float(df['revenue'].sum())
    avg_order_value = float(df['revenue'].sum() / df['quantity_sold'].sum())
    
    # Category analysis - top 10 only
    category_stats = df.groupby('category').agg({
        'quantity_sold': 'sum',
        'revenue': 'sum'
    }).sort_values('revenue', ascending=False).head(10)
    
    # Brand analysis - top 10 only
    brand_stats = df.groupby('brand').agg({
        'quantity_sold': 'sum',
        'revenue': 'sum'
    }).sort_values('revenue', ascending=False).head(10)
    
    # Time series aggregation - last 30 days only for performance
    recent_cutoff = df['date'].max() - pd.Timedelta(days=30)
    recent_df = df[df['date'] >= recent_cutoff]
    daily_stats = recent_df.groupby('date').agg({
        'quantity_sold': 'sum',
        'revenue': 'sum'
    }).reset_index().sort_values('date')
    
    return {
        "summary": {
            "total_products": total_products,
            "total_sales": total_sales,
            "total_revenue": total_revenue,
            "avg_order_value": avg_order_value,
            "date_range": {
                "start": df['date'].min().isoformat(),
                "end": df['date'].max().isoformat()
            }
        },
        "category_performance": category_stats.to_dict('index'),
        "brand_performance": brand_stats.to_dict('index'),
        "daily_trends": daily_stats.to_dict('records')
    }


@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """Train the prediction model"""
    global model_trained
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset available for training")
    
    def train_model_task():
        global model_trained
        try:
            trend_predictor.fit(current_dataset)
            trend_predictor.save_models()
            model_trained = True
            print("Model training completed successfully")
        except Exception as e:
            print(f"Model training failed: {e}")
            model_trained = False
    
    background_tasks.add_task(train_model_task)
    
    return {
        "message": "Model training started in background",
        "status": "training",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/status")
async def get_model_status():
    """Get current model status and performance metrics"""
    if not model_trained:
        return {
            "trained": False,
            "message": "Model not trained yet"
        }
    
    try:
        metrics = {}
        if hasattr(trend_predictor, 'model_metrics'):
            metrics = {name: metric.dict() for name, metric in trend_predictor.model_metrics.items()}
        
        feature_importance = trend_predictor.get_feature_importance()
        
        return {
            "trained": True,
            "ensemble_weights": trend_predictor.ensemble_weights,
            "model_metrics": metrics,
            "feature_importance": feature_importance,
            "sequence_length": trend_predictor.sequence_length
        }
    except Exception as e:
        return {
            "trained": True,
            "error": f"Could not retrieve model details: {str(e)}"
        }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset for training"""
    global current_dataset, model_trained
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Basic validation
        required_columns = ['product_id', 'date', 'quantity_sold']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        current_dataset = df
        model_trained = False  # Reset model training status
        
        return {
            "message": "Dataset uploaded successfully",
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "date_range": {
                "start": df['date'].min().isoformat(),
                "end": df['date'].max().isoformat()
            },
            "products": df['product_id'].nunique()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/generate-sample-data")
async def generate_sample_data(
    n_products: int = 20,
    days: int = 365
):
    """Load sample data from Superstore dataset"""
    global current_dataset, model_trained
    
    try:
        loader = SuperstoreDataLoader()
        current_dataset = loader.get_sample_data(n_products=n_products, days=days)
        
        model_trained = False  # Reset model training status
        
        return {
            "message": "Superstore data loaded successfully",
            "shape": current_dataset.shape,
            "products": n_products,
            "days": days,
            "date_range": {
                "start": current_dataset['date'].min().isoformat(),
                "end": current_dataset['date'].max().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Superstore data: {str(e)}")


@app.get("/reload-superstore-data")
async def reload_superstore_data(n_products: int = None, days: int = 365):
    """Force reload Superstore data and refresh dashboard"""
    global current_dataset, model_trained
    
    try:
        loader = SuperstoreDataLoader()
        
        if n_products is not None:
            # Load sample dataset
            current_dataset = loader.get_sample_data(n_products=n_products, days=days)
            message = f"Sample Superstore dataset reloaded successfully ({n_products} products)"
        else:
            # Load complete dataset
            current_dataset = loader.get_complete_dataset()
            message = "Complete Superstore dataset reloaded successfully"
        
        model_trained = False  # Reset model training status
        
        return {
            "message": message,
            "shape": current_dataset.shape,
            "date_range": {
                "start": current_dataset['date'].min().isoformat(),
                "end": current_dataset['date'].max().isoformat()
            },
            "dataset_info": {
                "total_products": current_dataset['product_id'].nunique(),
                "total_sales": int(current_dataset['quantity_sold'].sum()),
                "total_revenue": float(current_dataset['revenue'].sum()),
                "categories": current_dataset['category'].nunique()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload Superstore data: {str(e)}")





@app.get("/download-sample-data")
async def download_sample_data():
    """Download current dataset as CSV"""
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset available")
    
    # Save to temporary file
    filename = f"ecommerce_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = f"temp_{filename}"
    
    try:
        current_dataset.to_csv(filepath, index=False)
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='text/csv'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create download: {str(e)}")


@app.get("/alerts")
async def get_trend_alerts():
    """Get trend alerts for products with significant changes"""
    if current_dataset is None or not model_trained:
        return {"alerts": []}
    
    alerts = []
    
    try:
        # Simple alert logic - find products with significant recent changes
        recent_data = current_dataset[
            current_dataset['date'] >= (current_dataset['date'].max() - pd.Timedelta(days=7))
        ]
        
        for product_id in recent_data['product_id'].unique():
            product_recent = recent_data[recent_data['product_id'] == product_id]
            product_historical = current_dataset[
                (current_dataset['product_id'] == product_id) &
                (current_dataset['date'] < (current_dataset['date'].max() - pd.Timedelta(days=7)))
            ]
            
            if len(product_recent) > 0 and len(product_historical) > 0:
                recent_avg = product_recent['quantity_sold'].mean()
                historical_avg = product_historical['quantity_sold'].mean()
                
                if historical_avg > 0:
                    change_pct = (recent_avg - historical_avg) / historical_avg * 100
                    
                    if abs(change_pct) > 50:  # Alert if change > 50%
                        alert_type = "spike" if change_pct > 0 else "drop"
                        severity = "high" if abs(change_pct) > 100 else "medium"
                        
                        alerts.append({
                            "product_id": product_id,
                            "alert_type": alert_type,
                            "severity": severity,
                            "message": f"Sales {alert_type} detected: {change_pct:.1f}% change from historical average",
                            "confidence": min(1.0, abs(change_pct) / 100),
                            "detected_at": datetime.now().isoformat()
                        })
        
        return {"alerts": alerts}
        
    except Exception as e:
        return {"alerts": [], "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)