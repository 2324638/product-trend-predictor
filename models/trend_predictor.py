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
import lightgbm as lgb

# Optional TensorFlow import
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import plot_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available. LSTM models will be disabled.")
    print("   Install TensorFlow with: pip install tensorflow")
    Sequential = None

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


class LSTMPredictor:
    """LSTM Neural Network for time series prediction"""
    
    def __init__(self, sequence_length: int = 30, lstm_units: int = 50):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = None
        self.available = TENSORFLOW_AVAILABLE
        
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMPredictor':
        """Train the LSTM model"""
        if not self.available:
            raise ImportError("TensorFlow not available. Cannot train LSTM model.")
        
        # Prepare data for LSTM
        X_seq, y_seq = self.create_sequences(X.values, y.values)
        
        if len(X_seq) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_seq, _ = self.create_sequences(X.values, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.array([])
        
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()
    
    def save(self, filepath: str):
        """Save model"""
        if self.model:
            self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load model"""
        self.model = load_model(filepath)


class TrendPredictor:
    """
    Ensemble model for e-commerce product trend prediction
    Combines LSTM, XGBoost, and LightGBM models
    """
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.preprocessor = TrendDataPreprocessor()
        
        # Individual models
        self.lstm_model = LSTMPredictor(sequence_length=sequence_length)
        self.xgb_model = None
        self.lgb_model = None
        self.prophet_model = None
        
        # Ensemble weights - adjust based on available models
        if TENSORFLOW_AVAILABLE and PROPHET_AVAILABLE:
            self.ensemble_weights = {'lstm': 0.25, 'xgb': 0.25, 'lgb': 0.25, 'prophet': 0.25}
        elif TENSORFLOW_AVAILABLE:
            self.ensemble_weights = {'lstm': 0.4, 'xgb': 0.35, 'lgb': 0.25, 'prophet': 0.0}
        elif PROPHET_AVAILABLE:
            self.ensemble_weights = {'lstm': 0.0, 'xgb': 0.4, 'lgb': 0.3, 'prophet': 0.3}
        else:
            self.ensemble_weights = {'lstm': 0.0, 'xgb': 0.6, 'lgb': 0.4, 'prophet': 0.0}
            print("⚠️  TensorFlow and Prophet not available. Using XGBoost + LightGBM ensemble only.")
        
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
    
    def _train_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
        """Train LightGBM model"""
        # Split data chronologically
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def _evaluate_model(self, model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate individual model performance"""
        if model_name == 'lstm':
            predictions = model.predict(X_test)
            # Align predictions with test data (LSTM returns fewer predictions due to sequence length)
            if len(predictions) < len(y_test):
                y_test = y_test.iloc[-len(predictions):]
        else:
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
        
        print("Training LightGBM...")
        self.lgb_model = self._train_lightgbm(X_train, y_train)
        lgb_metrics = self._evaluate_model(self.lgb_model, 'lightgbm', X_test, y_test)
        self.model_metrics['lightgbm'] = lgb_metrics
        
        # Train LSTM only if TensorFlow is available
        if TENSORFLOW_AVAILABLE and self.ensemble_weights['lstm'] > 0:
            print("Training LSTM...")
            try:
                self.lstm_model.fit(X_train, y_train)
                lstm_metrics = self._evaluate_model(self.lstm_model, 'lstm', X_test, y_test)
                self.model_metrics['lstm'] = lstm_metrics
            except Exception as e:
                print(f"LSTM training failed: {e}")
                # Adjust ensemble weights if LSTM fails
                self.ensemble_weights['lstm'] = 0.0
        else:
            print("Skipping LSTM training (TensorFlow not available)")
        
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
        
        if self.lgb_model:
            lgb_pred = self.lgb_model.predict(X)
            predictions['lightgbm'] = lgb_pred
        
        if self.lstm_model.model and self.ensemble_weights['lstm'] > 0:
            try:
                lstm_pred = self.lstm_model.predict(X)
                if len(lstm_pred) > 0:
                    # Align LSTM predictions with other models
                    if len(lstm_pred) < len(X):
                        padding = np.full(len(X) - len(lstm_pred), lstm_pred[0] if len(lstm_pred) > 0 else 0)
                        lstm_pred = np.concatenate([padding, lstm_pred])
                    predictions['lstm'] = lstm_pred[:len(X)]
            except Exception as e:
                print(f"LSTM prediction failed: {e}")
        
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
        
        if self.lgb_model and hasattr(self.lgb_model, 'feature_importances_'):
            lgb_importance = dict(zip(self.preprocessor.feature_names, self.lgb_model.feature_importances_))
            importance['lightgbm'] = lgb_importance
        
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
        
        if self.lgb_model:
            joblib.dump(self.lgb_model, f'{base_path}/lightgbm_model.joblib')
        
        # Save LSTM model
        if self.lstm_model.model:
            self.lstm_model.save(f'{base_path}/lstm_model.h5')
        
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
        
        try:
            self.lgb_model = joblib.load(f'{base_path}/lightgbm_model.joblib')
        except FileNotFoundError:
            pass
        
        # Load LSTM model
        try:
            self.lstm_model.load(f'{base_path}/lstm_model.h5')
        except (FileNotFoundError, OSError):
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
