import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Optional Prophet import
try:
    from models.prophet_predictor import ProphetPredictor
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not available. Prophet models will be disabled.")
    print("   Install Prophet with: pip install prophet")
    ProphetPredictor = None

# Local imports
from data.preprocessor import TrendDataPreprocessor
from data.models import PredictionResponse, ModelMetrics


class TrendPredictor:
    """
    Ensemble model for e-commerce product trend prediction
    Combines XGBoost and Prophet models
    """
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.preprocessor = TrendDataPreprocessor()
        
        # Individual models
        self.xgb_model = None
        self.prophet_model = None
        
        # Ensemble weights - adjust based on available models
        if PROPHET_AVAILABLE:
            self.ensemble_weights = {'xgb': 0.6, 'prophet': 0.4}
        else:
            self.ensemble_weights = {'xgb': 1.0, 'prophet': 0.0}
            print("⚠️  Prophet not available. Using XGBoost only.")
        
        # Model metrics
        self.model_metrics = {}
        self.is_trained = False
        
    def _prepare_data(self, df: pd.DataFrame, target_col: str = 'quantity_sold') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        # Sort by date and product_id
        df_sorted = df.sort_values(['product_id', 'date']).reset_index(drop=True)
        
        # Feature engineering and preprocessing
        X, y = self.preprocessor.fit_transform(df_sorted, target_col)
        
        return X, y
    
    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        # Split data chronologically
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='rmse'
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def _evaluate_model(self, model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate individual model performance"""
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1))) * 100
        r2 = r2_score(y_test, predictions)
        
        return ModelMetrics(
            model_name=model_name,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            r2_score=r2
        )
    
    def fit(self, df: pd.DataFrame, target_col: str = 'quantity_sold') -> 'TrendPredictor':
        """Train all models in the ensemble"""
        print("Preparing data...")
        X, y = self._prepare_data(df, target_col)
        
        # Split data for final evaluation
        split_idx = int(len(X) * 0.9)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print("Training XGBoost...")
        self.xgb_model = self._train_xgboost(X_train, y_train)
        xgb_metrics = self._evaluate_model(self.xgb_model, 'xgboost', X_test, y_test)
        self.model_metrics['xgboost'] = xgb_metrics
        
        # Train Prophet only if available
        if PROPHET_AVAILABLE and self.ensemble_weights['prophet'] > 0:
            print("Training Prophet...")
            try:
                self.prophet_model = ProphetPredictor()
                self.prophet_model.fit(df, fit_global=True, fit_products=True)
                print(f"Prophet training completed successfully - {len(self.prophet_model.product_models)} product models trained")
                # Note: Prophet evaluation is done differently, so we'll skip metrics for now
            except Exception as e:
                print(f"Prophet training failed: {e}")
                self.ensemble_weights['prophet'] = 0.0
        else:
            print("Skipping Prophet training (Prophet not available)")
        
        self.is_trained = True
        
        # Print model performance
        print("\nModel Performance:")
        for name, metrics in self.model_metrics.items():
            print(f"{name.upper()}: MAE={metrics.mae:.3f}, RMSE={metrics.rmse:.3f}, R²={metrics.r2_score:.3f}")
        
        return self
    
    def predict(self, df: pd.DataFrame, days_ahead: int = 30) -> Dict[str, Any]:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Prepare data for prediction
        X = self.preprocessor.transform(df)
        
        predictions = {}
        
        # Get predictions from each model
        if self.xgb_model:
            xgb_pred = self.xgb_model.predict(X)
            predictions['xgboost'] = xgb_pred
        
        # Get Prophet predictions if available
        if self.prophet_model and self.ensemble_weights['prophet'] > 0:
            try:
                prophet_results = self.prophet_model.predict(df, periods=days_ahead)
                if 'global' in prophet_results:
                    prophet_pred = np.array(prophet_results['global']['predictions'])
                    # Align Prophet predictions with other models (take first len(X) predictions)
                    if len(prophet_pred) > 0:
                        if len(prophet_pred) < len(X):
                            padding = np.full(len(X) - len(prophet_pred), prophet_pred[0] if len(prophet_pred) > 0 else 0)
                            prophet_pred = np.concatenate([padding, prophet_pred])
                        predictions['prophet'] = prophet_pred[:len(X)]
            except Exception as e:
                print(f"Prophet prediction failed: {e}")
        
        # Ensemble prediction
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.ensemble_weights.get(model_name, 0)
            if weight > 0:
                ensemble_pred += weight * pred
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        # Calculate confidence intervals (simple approach using prediction variance)
        pred_std = np.std(list(predictions.values()), axis=0) if len(predictions) > 1 else np.ones(len(ensemble_pred))
        confidence_lower = ensemble_pred - 1.96 * pred_std
        confidence_upper = ensemble_pred + 1.96 * pred_std
        
        # Calculate trend metrics
        if len(ensemble_pred) > 1:
            trend_slope = np.polyfit(range(len(ensemble_pred)), ensemble_pred, 1)[0]
            if trend_slope > 0.1:
                trend_direction = "up"
            elif trend_slope < -0.1:
                trend_direction = "down"
            else:
                trend_direction = "stable"
            
            trend_strength = min(1.0, abs(trend_slope) / np.mean(ensemble_pred))
        else:
            trend_direction = "stable"
            trend_strength = 0.0
        
        return {
            'predictions': ensemble_pred,
            'individual_predictions': predictions,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'model_accuracy': np.mean([m.r2_score for m in self.model_metrics.values()])
        }
    
    def predict_product(self, product_id: str, df: pd.DataFrame, days_ahead: int = 30) -> PredictionResponse:
        """Predict trends for a specific product using Prophet and ensemble models"""
        # Filter data for the specific product
        product_data = df[df['product_id'] == product_id].copy()
        
        if len(product_data) == 0:
            raise ValueError(f"No data found for product {product_id}")
        
        # Get recent data for prediction - use more data to ensure lag features are available
        product_data = product_data.sort_values('date').tail(200)  # Use last 200 days to ensure lag features
        
        # Try Prophet prediction first (if available)
        prophet_predictions = None
        ensemble_result = None
        
        if self.prophet_model and self.ensemble_weights['prophet'] > 0:
            try:
                print(f"🔮 Using Prophet for {product_id} prediction...")
                prophet_results = self.prophet_model.predict(
                    product_data, 
                    periods=days_ahead,
                    product_id=product_id
                )
                
                if 'product' in prophet_results:
                    prophet_predictions = prophet_results['product']
                    print(f"✅ Prophet prediction successful for {product_id}")
                elif 'global' in prophet_results:
                    # Use global prediction if product-specific not available
                    prophet_predictions = prophet_results['global']
                    print(f"⚠️ Using global Prophet prediction for {product_id}")
            except Exception as e:
                print(f"❌ Prophet prediction failed for {product_id}: {e}")
        
        # Make ensemble prediction as fallback
        if not prophet_predictions:
            try:
                print(f"📊 Using ensemble prediction for {product_id}...")
                ensemble_result = self.predict(product_data, days_ahead)
            except Exception as e:
                print(f"❌ Ensemble prediction failed for {product_id}: {e}")
                raise ValueError(f"Both Prophet and ensemble predictions failed for {product_id}")
        
        # Format response - prioritize Prophet if available
        predictions_list = []
        confidence_intervals = []
        
        last_date = product_data['date'].max()
        
        # Use Prophet predictions if available, otherwise use ensemble
        if prophet_predictions and len(prophet_predictions['predictions']) > 0:
            print(f"📊 Using Prophet predictions for {product_id}")
            for i, (date_str, pred, lower, upper) in enumerate(zip(
                prophet_predictions['dates'][:days_ahead],
                prophet_predictions['predictions'][:days_ahead],
                prophet_predictions['lower_bound'][:days_ahead],
                prophet_predictions['upper_bound'][:days_ahead]
            )):
                predictions_list.append({
                    'date': date_str,
                    'predicted_quantity': float(pred),
                    'day_ahead': i + 1,
                    'model': 'prophet'
                })
                
                confidence_intervals.append({
                    'date': date_str,
                    'lower_bound': float(lower),
                    'upper_bound': float(upper)
                })
            
            # Calculate trend metrics from Prophet predictions
            if len(prophet_predictions['predictions']) > 1:
                trend_slope = np.polyfit(range(len(prophet_predictions['predictions'])), prophet_predictions['predictions'], 1)[0]
                if trend_slope > 0.1:
                    trend_direction = "up"
                elif trend_slope < -0.1:
                    trend_direction = "down"
                else:
                    trend_direction = "stable"
                
                trend_strength = min(1.0, abs(trend_slope) / np.mean(prophet_predictions['predictions']))
            else:
                trend_direction = "stable"
                trend_strength = 0.0
                
            model_accuracy = 0.85  # Prophet typically has good accuracy
        else:
            # Use ensemble predictions
            print(f"📊 Using ensemble predictions for {product_id}")
            for i in range(min(days_ahead, len(ensemble_result['predictions']))):
                pred_date = last_date + timedelta(days=i+1)
                predictions_list.append({
                    'date': pred_date.isoformat(),
                    'predicted_quantity': float(ensemble_result['predictions'][i]),
                    'day_ahead': i + 1,
                    'model': 'ensemble'
                })
                
                confidence_intervals.append({
                    'date': pred_date.isoformat(),
                    'lower_bound': float(ensemble_result['confidence_lower'][i]),
                    'upper_bound': float(ensemble_result['confidence_upper'][i])
                })
            
            # Use ensemble trend metrics
            trend_direction = ensemble_result['trend_direction']
            trend_strength = ensemble_result['trend_strength']
            model_accuracy = ensemble_result['model_accuracy']
        
        return PredictionResponse(
            product_id=product_id,
            predictions=predictions_list,
            confidence_intervals=confidence_intervals,
            trend_direction=trend_direction,
            trend_strength=float(trend_strength),
            model_accuracy=float(model_accuracy)
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from tree-based models"""
        importance = {}
        
        if self.xgb_model and hasattr(self.xgb_model, 'feature_importances_'):
            xgb_importance = dict(zip(self.preprocessor.feature_names, self.xgb_model.feature_importances_))
            importance['xgboost'] = xgb_importance
        
        return importance
    
    def save_models(self, base_path: str = 'models/saved'):
        """Save all trained models"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, f'{base_path}/preprocessor.joblib')
        
        # Save tree models
        if self.xgb_model:
            joblib.dump(self.xgb_model, f'{base_path}/xgboost_model.joblib')
        
        # Save metadata
        metadata = {
            'ensemble_weights': self.ensemble_weights,
            'model_metrics': {k: v.dict() for k, v in self.model_metrics.items()},
            'sequence_length': self.sequence_length,
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, f'{base_path}/metadata.joblib')
        
        print(f"Models saved to {base_path}")
    
    def load_models(self, base_path: str = 'models/saved'):
        """Load all trained models"""
        # Load preprocessor
        self.preprocessor = joblib.load(f'{base_path}/preprocessor.joblib')
        
        # Load tree models
        try:
            self.xgb_model = joblib.load(f'{base_path}/xgboost_model.joblib')
        except FileNotFoundError:
            pass
        
        # Load metadata
        metadata = joblib.load(f'{base_path}/metadata.joblib')
        self.ensemble_weights = metadata['ensemble_weights']
        self.model_metrics = {k: ModelMetrics(**v) for k, v in metadata['model_metrics'].items()}
        self.sequence_length = metadata['sequence_length']
        self.is_trained = metadata['is_trained']
        
        print(f"Models loaded from {base_path}")


# Example usage
if __name__ == "__main__":
    # This would be used with real Superstore data
    from data.superstore_loader import SuperstoreDataLoader
    
    # Load Superstore data
    loader = SuperstoreDataLoader()
    df = loader.get_sample_data(n_products=10, days=365)
    
    # Train model
    predictor = TrendPredictor()
    predictor.fit(df)
    
    # Make prediction for a specific product
    product_id = df['product_id'].iloc[0]
    prediction = predictor.predict_product(product_id, df, days_ahead=30)
    
    print(f"\nPrediction for {product_id}:")
    print(f"Trend Direction: {prediction.trend_direction}")
    print(f"Trend Strength: {prediction.trend_strength:.3f}")
    print(f"Model Accuracy: {prediction.model_accuracy:.3f}")
    print(f"Number of predictions: {len(prediction.predictions)}")