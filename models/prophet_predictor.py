import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not available. Install with: pip install prophet")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class ProphetPredictor:
    """
    Prophet-based time series predictor for product trend forecasting
    """
    
    def __init__(self, 
                 daily_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 yearly_seasonality: bool = True,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0):
        """
        Initialize Prophet predictor
        
        Args:
            daily_seasonality: Whether to include daily seasonality
            weekly_seasonality: Whether to include weekly seasonality  
            yearly_seasonality: Whether to include yearly seasonality
            changepoint_prior_scale: Flexibility of the trend
            seasonality_prior_scale: Flexibility of the seasonality
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")
        
        self.model = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        
        self.is_fitted = False
        self.product_models = {}  # Store models for each product
        self.global_model = None  # Store global model for overall trends
        
    def prepare_data(self, df: pd.DataFrame, 
                    date_col: str = 'date',
                    value_col: str = 'quantity_sold',
                    product_col: str = 'product_id') -> pd.DataFrame:
        """
        Prepare data for Prophet (aggregate by date)
        
        Args:
            df: Input dataframe
            date_col: Name of date column
            value_col: Name of value column to forecast
            product_col: Name of product column
            
        Returns:
            DataFrame with columns ['ds', 'y'] for Prophet
        """
        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Aggregate by date and product
        ts_df = df.groupby([date_col, product_col])[value_col].sum().reset_index()
        
        # Rename columns for Prophet
        ts_df = ts_df.rename(columns={date_col: 'ds', value_col: 'y'})
        
        return ts_df
    
    def fit(self, df: pd.DataFrame, 
            date_col: str = 'date',
            value_col: str = 'quantity_sold',
            product_col: str = 'product_id',
            fit_global: bool = True,
            fit_products: bool = True) -> 'ProphetPredictor':
        """
        Fit Prophet model(s)
        
        Args:
            df: Input dataframe
            date_col: Name of date column
            value_col: Name of value column to forecast
            product_col: Name of product column
            fit_global: Whether to fit a global model for overall trends
            fit_products: Whether to fit individual models for each product
        """
        print("🔮 Training Prophet models...")
        
        # Prepare data
        ts_df = self.prepare_data(df, date_col, value_col, product_col)
        
        # Fit global model (all products combined)
        if fit_global:
            print("   Training global model...")
            global_data = ts_df.groupby('ds')['y'].sum().reset_index()
            self.global_model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            self.global_model.fit(global_data)
        
        # Fit individual product models
        if fit_products:
            print("   Training product-specific models...")
            for product_id in ts_df[product_col].unique():
                product_data = ts_df[ts_df[product_col] == product_id].copy()
                if len(product_data) >= 10:  # Need minimum data points
                    try:
                        model = Prophet(
                            daily_seasonality=True,
                            weekly_seasonality=True,
                            yearly_seasonality=True
                        )
                        model.fit(product_data[['ds', 'y']])
                        self.product_models[product_id] = model
                    except Exception as e:
                        print(f"   ⚠️ Could not fit model for {product_id}: {e}")
        
        self.is_fitted = True
        print(f"✅ Prophet training complete! Trained {len(self.product_models)} product models")
        
        return self
    
    def predict(self, df: pd.DataFrame,
                periods: int = 30,
                date_col: str = 'date',
                value_col: str = 'quantity_sold',
                product_col: str = 'product_id',
                product_id: Optional[str] = None) -> Dict:
        """
        Make predictions using Prophet
        
        Args:
            df: Input dataframe
            periods: Number of periods to forecast
            date_col: Name of date column
            value_col: Name of value column
            product_col: Name of product column
            product_id: Specific product to predict (if None, predicts for all)
            
        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        results = {}
        
        # Prepare data
        ts_df = self.prepare_data(df, date_col, value_col, product_col)
        
        # Global prediction
        if self.global_model:
            print("🔮 Making global predictions...")
            future_global = self.global_model.make_future_dataframe(periods=periods)
            forecast_global = self.global_model.predict(future_global)
            
            # Get only future predictions
            last_date = ts_df['ds'].max()
            future_forecast = forecast_global[forecast_global['ds'] > last_date].copy()
            
            results['global'] = {
                'dates': future_forecast['ds'].tolist(),
                'predictions': future_forecast['yhat'].tolist(),
                'lower_bound': future_forecast['yhat_lower'].tolist(),
                'upper_bound': future_forecast['yhat_upper'].tolist(),
                'trend': future_forecast['trend'].tolist()
            }
        
        # Product-specific predictions
        if product_id and product_id in self.product_models:
            print(f"🔮 Making predictions for {product_id}...")
            product_data = ts_df[ts_df[product_col] == product_id].copy()
            
            if len(product_data) > 0:
                future_product = self.product_models[product_id].make_future_dataframe(periods=periods)
                forecast_product = self.product_models[product_id].predict(future_product)
                
                # Get only future predictions
                last_date = product_data['ds'].max()
                future_forecast = forecast_product[forecast_product['ds'] > last_date].copy()
                
                results['product'] = {
                    'product_id': product_id,
                    'dates': future_forecast['ds'].tolist(),
                    'predictions': future_forecast['yhat'].tolist(),
                    'lower_bound': future_forecast['yhat_lower'].tolist(),
                    'upper_bound': future_forecast['yhat_upper'].tolist(),
                    'trend': future_forecast['trend'].tolist()
                }
        
        elif not product_id:
            # Predict for all products
            print("🔮 Making predictions for all products...")
            product_predictions = {}
            
            for pid in self.product_models.keys():
                try:
                    product_data = ts_df[ts_df[product_col] == pid].copy()
                    if len(product_data) > 0:
                        future_product = self.product_models[pid].make_future_dataframe(periods=periods)
                        forecast_product = self.product_models[pid].predict(future_product)
                        
                        # Get only future predictions
                        last_date = product_data['ds'].max()
                        future_forecast = forecast_product[forecast_product['ds'] > last_date].copy()
                        
                        product_predictions[pid] = {
                            'dates': future_forecast['ds'].tolist(),
                            'predictions': future_forecast['yhat'].tolist(),
                            'lower_bound': future_forecast['yhat_lower'].tolist(),
                            'upper_bound': future_forecast['yhat_upper'].tolist(),
                            'trend': future_forecast['trend'].tolist()
                        }
                except Exception as e:
                    print(f"   ⚠️ Error predicting for {pid}: {e}")
            
            results['products'] = product_predictions
        
        return results
    
    def plot_forecast(self, df: pd.DataFrame,
                     product_id: Optional[str] = None,
                     periods: int = 30,
                     date_col: str = 'date',
                     value_col: str = 'quantity_sold',
                     product_col: str = 'product_id',
                     save_path: Optional[str] = None):
        """
        Plot Prophet forecast
        
        Args:
            df: Input dataframe
            product_id: Specific product to plot
            periods: Number of periods to forecast
            date_col: Name of date column
            value_col: Name of value column
            product_col: Name of product column
            save_path: Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            print("⚠️ Matplotlib not available for plotting")
            return
        
        if not self.is_fitted:
            print("⚠️ Model must be fitted before plotting")
            return
        
        # Prepare data
        ts_df = self.prepare_data(df, date_col, value_col, product_col)
        
        if product_id and product_id in self.product_models:
            # Plot specific product
            product_data = ts_df[ts_df[product_col] == product_id].copy()
            if len(product_data) > 0:
                future = self.product_models[product_id].make_future_dataframe(periods=periods)
                forecast = self.product_models[product_id].predict(future)
                
                fig = self.product_models[product_id].plot(forecast)
                plt.title(f"Prophet Forecast for {product_id}")
                plt.xlabel("Date")
                plt.ylabel("Quantity Sold")
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
        
        elif self.global_model:
            # Plot global forecast
            global_data = ts_df.groupby('ds')['y'].sum().reset_index()
            future = self.global_model.make_future_dataframe(periods=periods)
            forecast = self.global_model.predict(future)
            
            fig = self.global_model.plot(forecast)
            plt.title("Prophet Global Sales Forecast")
            plt.xlabel("Date")
            plt.ylabel("Total Sales")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def get_components(self, product_id: Optional[str] = None, save_path: Optional[str] = None):
        """
        Plot Prophet components (trend, seasonality)
        
        Args:
            product_id: Specific product to analyze
            save_path: Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            print("⚠️ Matplotlib not available for plotting")
            return
        
        if not self.is_fitted:
            print("⚠️ Model must be fitted before plotting components")
            return
        
        if product_id and product_id in self.product_models:
            # Plot components for specific product
            fig = self.product_models[product_id].plot_components()
            plt.title(f"Prophet Components for {product_id}")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        elif self.global_model:
            # Plot global components
            fig = self.global_model.plot_components()
            plt.title("Prophet Global Components")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_model(self, filepath: str):
        """Save the Prophet model"""
        import joblib
        joblib.dump(self, filepath)
        print(f"✅ Prophet model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the Prophet model"""
        import joblib
        loaded_model = joblib.load(filepath)
        self.model = loaded_model.model
        self.is_fitted = loaded_model.is_fitted
        self.product_models = loaded_model.product_models
        self.global_model = loaded_model.global_model
        print(f"✅ Prophet model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # This would be used with real Superstore data
    from data.superstore_loader import SuperstoreDataLoader
    
    # Load Superstore data
    loader = SuperstoreDataLoader()
    df = loader.get_sample_data(n_products=5, days=365)
    
    # Initialize and train Prophet
    prophet_predictor = ProphetPredictor()
    prophet_predictor.fit(df)
    
    # Make predictions
    predictions = prophet_predictor.predict(df, periods=30)
    
    print("📊 Prophet Predictions:")
    if 'global' in predictions:
        global_pred = predictions['global']
        avg_prediction = np.mean(global_pred['predictions'])
        print(f"   Global average prediction: {avg_prediction:.2f}")
    
    if 'products' in predictions:
        for product_id, pred in predictions['products'].items():
            avg_pred = np.mean(pred['predictions'])
            print(f"   {product_id}: {avg_pred:.2f}") 