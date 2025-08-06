import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


class TrendDataPreprocessor:
    """
    Advanced data preprocessing and feature engineering for trend prediction
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Extract comprehensive temporal features from date column"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['quarter'] = df[date_col].dt.quarter
        
        # Boolean temporal features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Season encoding
        season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                     3: 'spring', 4: 'spring', 5: 'spring',
                     6: 'summer', 7: 'summer', 8: 'summer',
                     9: 'fall', 10: 'fall', 11: 'fall'}
        df['season'] = df['month'].map(season_map)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                          lags: List[int] = [1, 2, 3, 7, 14, 30]) -> pd.DataFrame:
        """Create lagged features for time series analysis"""
        df = df.copy()
        df = df.sort_values('date')
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby('product_id')[target_col].shift(lag)
            
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str,
                               windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """Create rolling statistical features"""
        df = df.copy()
        df = df.sort_values('date')
        
        for window in windows:
            # Rolling statistics
            df[f'{target_col}_rolling_mean_{window}'] = (
                df.groupby('product_id')[target_col]
                .rolling(window=window, min_periods=1)
                .mean().reset_index(0, drop=True)
            )
            
            df[f'{target_col}_rolling_std_{window}'] = (
                df.groupby('product_id')[target_col]
                .rolling(window=window, min_periods=1)
                .std().reset_index(0, drop=True)
            )
            
            df[f'{target_col}_rolling_max_{window}'] = (
                df.groupby('product_id')[target_col]
                .rolling(window=window, min_periods=1)
                .max().reset_index(0, drop=True)
            )
            
            df[f'{target_col}_rolling_min_{window}'] = (
                df.groupby('product_id')[target_col]
                .rolling(window=window, min_periods=1)
                .min().reset_index(0, drop=True)
            )
            
        return df
    
    def create_trend_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create trend and momentum features"""
        df = df.copy()
        df = df.sort_values('date')
        
        # Price change features
        df[f'{target_col}_pct_change_1d'] = (
            df.groupby('product_id')[target_col].pct_change(1)
        )
        df[f'{target_col}_pct_change_7d'] = (
            df.groupby('product_id')[target_col].pct_change(7)
        )
        df[f'{target_col}_pct_change_30d'] = (
            df.groupby('product_id')[target_col].pct_change(30)
        )
        
        # Momentum indicators
        df[f'{target_col}_momentum_3d'] = (
            df.groupby('product_id')[target_col].diff(3)
        )
        df[f'{target_col}_momentum_7d'] = (
            df.groupby('product_id')[target_col].diff(7)
        )
        
        # Volatility
        df[f'{target_col}_volatility_7d'] = (
            df.groupby('product_id')[f'{target_col}_pct_change_1d']
            .rolling(window=7, min_periods=1)
            .std().reset_index(0, drop=True)
        )
        
        return df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        categorical_cols = ['category', 'brand', 'channel', 'season']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    known_classes = set(self.encoders[col].classes_)
                    df[f'{col}_temp'] = df[col].astype(str)
                    df.loc[~df[f'{col}_temp'].isin(known_classes), f'{col}_temp'] = 'unknown'
                    
                    if 'unknown' not in known_classes:
                        # Add unknown category to encoder
                        all_classes = list(known_classes) + ['unknown']
                        self.encoders[col].classes_ = np.array(all_classes)
                    
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[f'{col}_temp'])
                    df.drop(f'{col}_temp', axis=1, inplace=True)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Price-related interactions
        if 'price' in df.columns and 'discount_applied' in df.columns:
            df['effective_price'] = df['price'] * (1 - df['discount_applied'] / 100)
            df['discount_impact'] = df['price'] - df['effective_price']
        
        # Seasonal-price interactions
        if 'month' in df.columns and 'price' in df.columns:
            df['price_season_interaction'] = df['price'] * df['month']
        
        # Weekend-discount interaction
        if 'is_weekend' in df.columns and 'discount_applied' in df.columns:
            df['weekend_discount'] = df['is_weekend'] * df['discount_applied']
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using domain-appropriate strategies"""
        df = df.copy()
        
        # Forward fill for time series data
        time_series_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['lag', 'rolling', 'momentum', 'pct_change']
        )]
        for col in time_series_cols:
            if df[col].isnull().sum() > 0:
                # Use newer pandas method
                df[col] = df.groupby('product_id')[col].ffill()
                # Fill any remaining NaN with 0 for lag features
                df[col] = df[col].fillna(0)
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown', inplace=True)
        
        # Final check - fill any remaining NaN with 0 for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(0, inplace=True)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        df = df.copy()
        
        # Select numerical columns for scaling
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['product_id', 'year', 'month', 'day', 'day_of_week', 
                       'day_of_year', 'week_of_year', 'quarter', 'is_weekend',
                       'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end']
        
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        
        if fit:
            self.scalers['standard'] = StandardScaler()
            df[cols_to_scale] = self.scalers['standard'].fit_transform(df[cols_to_scale])
        else:
            if 'standard' in self.scalers:
                df[cols_to_scale] = self.scalers['standard'].transform(df[cols_to_scale])
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> pd.DataFrame:
        """Select top k features using statistical tests"""
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X, y)
            self.feature_names = X.columns[self.feature_selector.get_support()].tolist()
        else:
            X_selected = self.feature_selector.transform(X)
        
        return pd.DataFrame(X_selected, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'quantity_sold') -> Tuple[pd.DataFrame, pd.Series]:
        """Complete preprocessing pipeline"""
        # Create all feature types
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.create_trend_features(df, target_col)
        df = self.create_categorical_features(df)
        df = self.create_interaction_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Prepare features and target - only include numerical columns
        exclude_cols = ['date', 'product_id', target_col, 'category', 'brand', 'channel', 'season', 
                       'sub_category', 'region', 'state', 'city', 'segment', 'ship_mode', 'customer_name', 'product_name']
        
        # Get only numerical columns for feature selection
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Scale features
        X = self.scale_features(X, fit=True)
        
        # Feature selection
        X = self.select_features(X, y)
        
        return X, y
    
    def transform(self, df: pd.DataFrame, target_col: str = 'quantity_sold') -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        # Create all feature types
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.create_trend_features(df, target_col)
        df = self.create_categorical_features(df)
        df = self.create_interaction_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Prepare features
        X = df[self.feature_names].copy()
        
        # Scale features
        X = self.scale_features(X, fit=False)
        
        return X
