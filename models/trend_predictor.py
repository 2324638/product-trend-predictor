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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

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
        
        # Ensemble weights
        self.ensemble_weights = {'lstm': 0.4, 'xgb': 0.35, 'lgb': 0.25}
        
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
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
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
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
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
        
        print("Training LSTM...")
        try:
            self.lstm_model.fit(X_train, y_train)
            lstm_metrics = self._evaluate_model(self.lstm_model, 'lstm', X_test, y_test)
            self.model_metrics['lstm'] = lstm_metrics
        except Exception as e:
            print(f"LSTM training failed: {e}")
            # Adjust ensemble weights if LSTM fails
            self.ensemble_weights = {'lstm': 0.0, 'xgb': 0.6, 'lgb': 0.4}
        
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
        """Predict trends for a specific product"""
        # Filter data for the specific product
        product_data = df[df['product_id'] == product_id].copy()
        
        if len(product_data) == 0:
            raise ValueError(f"No data found for product {product_id}")
        
        # Get recent data for prediction
        product_data = product_data.sort_values('date').tail(100)  # Use last 100 days
        
        # Make prediction
        result = self.predict(product_data, days_ahead)
        
        # Format response
        predictions_list = []
        confidence_intervals = []
        
        last_date = product_data['date'].max()
        
        for i in range(min(days_ahead, len(result['predictions']))):
            pred_date = last_date + timedelta(days=i+1)
            predictions_list.append({
                'date': pred_date.isoformat(),
                'predicted_quantity': float(result['predictions'][i]),
                'day_ahead': i + 1
            })
            
            confidence_intervals.append({
                'date': pred_date.isoformat(),
                'lower_bound': float(result['confidence_lower'][i]),
                'upper_bound': float(result['confidence_upper'][i])
            })
        
        return PredictionResponse(
            product_id=product_id,
            predictions=predictions_list,
            confidence_intervals=confidence_intervals,
            trend_direction=result['trend_direction'],
            trend_strength=float(result['trend_strength']),
            model_accuracy=float(result['model_accuracy'])
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
    # This would be used with real data
    from data.data_generator import EcommerceDataGenerator
    
    # Generate sample data
    generator = EcommerceDataGenerator(seed=42)
    df = generator.generate_complete_dataset(n_products=10, 
                                           start_date=datetime(2023, 1, 1),
                                           end_date=datetime(2023, 12, 31))
    
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