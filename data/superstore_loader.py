import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from data.models import ProductData, SalesData, MarketData, SeasonalData


class SuperstoreDataLoader:
    """
    Load and preprocess Superstore dataset for trend prediction model training and testing
    """
    
    def __init__(self, csv_path: str = "data/superstore.csv"):
        self.csv_path = csv_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the Superstore dataset"""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"✅ Loaded Superstore dataset: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"Superstore dataset not found at {self.csv_path}")
        except Exception as e:
            raise Exception(f"Error loading Superstore dataset: {e}")
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the Superstore data for trend prediction"""
        if self.data is None:
            self.load_data()
        
        df = self.data.copy()
        
        # Convert date columns to datetime
        date_columns = ['Order Date', 'Ship Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create product_id if not exists
        if 'Product ID' not in df.columns:
            df['Product ID'] = df['Product Name'].astype('category').cat.codes.astype(str).str.zfill(4)
            df['Product ID'] = 'PROD_' + df['Product ID']
        
        # Standardize column names
        column_mapping = {
            'Order Date': 'date',
            'Ship Date': 'ship_date',
            'Product ID': 'product_id',
            'Product Name': 'product_name',
            'Category': 'category',
            'Sub-Category': 'sub_category',
            'Sales': 'revenue',
            'Region': 'region',
            'State': 'state',
            'City': 'city',
            'Customer ID': 'customer_id',
            'Customer Name': 'customer_name',
            'Segment': 'segment',
            'Ship Mode': 'ship_mode'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Handle missing quantity_sold column
        if 'quantity_sold' not in df.columns:
            # Since Superstore dataset doesn't have quantity, we'll create it
            # Option 1: Assume quantity = 1 for each order line (most common)
            df['quantity_sold'] = 1
            
            # Option 2: Create synthetic quantity based on sales and category averages
            # Calculate average price per category and estimate quantity
            category_avg_prices = df.groupby('category')['revenue'].mean()
            df['estimated_price'] = df['category'].map(category_avg_prices)
            df['quantity_sold'] = np.maximum(1, np.round(df['revenue'] / df['estimated_price']).astype(int))
            df = df.drop('estimated_price', axis=1)
        
        # Add missing columns that the model expects
        if 'price' not in df.columns:
            # Calculate average price per product
            df['price'] = df['revenue'] / df['quantity_sold']
        
        if 'brand' not in df.columns:
            df['brand'] = 'Superstore'
        
        if 'channel' not in df.columns:
            df['channel'] = 'online'
        
        if 'discount_applied' not in df.columns:
            df['discount_applied'] = 0.0
        
        if 'customer_count' not in df.columns:
            df['customer_count'] = 1
        
        # Add profit column if not exists (estimate based on category)
        if 'profit' not in df.columns:
            # Estimate profit margins by category (typical retail margins)
            profit_margins = {
                'Furniture': 0.15,
                'Office Supplies': 0.25,
                'Technology': 0.20
            }
            df['profit'] = df['revenue'] * df['category'].map(profit_margins).fillna(0.20)
        
        # Add temporal features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Add seasonal features
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Add holiday features (simplified)
        holidays = [
            (1, 1),   # New Year's Day
            (2, 14),  # Valentine's Day
            (3, 17),  # St. Patrick's Day
            (7, 4),   # Independence Day
            (10, 31), # Halloween
            (11, 25), # Thanksgiving (approximation)
            (12, 25), # Christmas
        ]
        df['is_holiday'] = df[['month', 'date']].apply(
            lambda x: (x['month'], x['date'].day) in holidays, axis=1
        ).astype(int)
        
        # Aggregate data by product and date for trend analysis
        # This creates daily sales data for each product
        daily_sales = df.groupby(['product_id', 'date']).agg({
            'product_name': 'first',  # Include product name
            'quantity_sold': 'sum',
            'revenue': 'sum',
            'profit': 'sum',
            'customer_count': 'sum',
            'price': 'mean',
            'category': 'first',
            'sub_category': 'first',
            'brand': 'first',
            'region': 'first',
            'state': 'first',
            'city': 'first',
            'segment': 'first',
            'ship_mode': 'first',
            'month': 'first',
            'quarter': 'first',
            'year': 'first',
            'day_of_week': 'first',
            'is_weekend': 'first',
            'day_of_year': 'first',
            'week_of_year': 'first',
            'season': 'first',
            'is_holiday': 'first'
        }).reset_index()
        
        # Add market simulation data (since Superstore doesn't have external market data)
        daily_sales = self._add_market_features(daily_sales)
        
        # Handle missing values
        daily_sales = self._handle_missing_values(daily_sales)
        
        print(f"✅ Preprocessed data shape: {daily_sales.shape}")
        print(f"✅ Date range: {daily_sales['date'].min()} to {daily_sales['date'].max()}")
        print(f"✅ Products: {daily_sales['product_id'].nunique()}")
        print(f"✅ Categories: {daily_sales['category'].nunique()}")
        
        return daily_sales
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simulated market features for trend prediction"""
        # Create a market index that varies over time
        dates = df['date'].unique().tolist()
        dates.sort()
        
        # Simulate market index with some trend and seasonality
        market_data = []
        base_index = 100.0
        
        for i, date in enumerate(dates):
            # Add trend
            trend = 1 + 0.0001 * i
            
            # Add seasonality
            seasonal = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            # Add some noise
            noise = np.random.normal(1.0, 0.02)
            
            market_index = base_index * trend * seasonal * noise
            market_index = max(50, min(200, market_index))
            
            market_data.append({
                'date': date,
                'market_index': round(market_index, 2),
                'competitor_price_avg': round(np.random.normal(50, 15), 2),
                'search_volume': max(0, int(np.random.normal(1000, 300))),
                'social_mentions': max(0, int(np.random.poisson(50))),
                'economic_indicator': round(np.random.normal(0.02, 0.01), 4)
            })
        
        market_df = pd.DataFrame(market_data)
        
        # Merge with main dataframe
        df = df.merge(market_df, on='date', how='left')
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the preprocessed data."""
        # Fill missing market_index with a default value
        df['market_index'] = df['market_index'].fillna(100.0)
        
        # Fill missing competitor_price_avg with a default value
        df['competitor_price_avg'] = df['competitor_price_avg'].fillna(50.0)
        
        # Fill missing search_volume with a default value
        df['search_volume'] = df['search_volume'].fillna(1000.0)
        
        # Fill missing social_mentions with a default value
        df['social_mentions'] = df['social_mentions'].fillna(50.0)
        
        # Fill missing economic_indicator with a default value
        df['economic_indicator'] = df['economic_indicator'].fillna(0.02)
        
        return df
    
    def get_sample_data(self, n_products: int = 20, days: int = 365) -> pd.DataFrame:
        """Get a sample of the data for demonstration"""
        df = self.preprocess_data()
        
        # Filter to recent data
        max_date = df['date'].max()
        start_date = max_date - timedelta(days=days)
        df_filtered = df[df['date'] >= start_date]
        
        # Select top products by revenue
        top_products = df_filtered.groupby('product_id')['revenue'].sum().nlargest(n_products).index
        df_sample = df_filtered[df_filtered['product_id'].isin(top_products)]
        
        print(f"✅ Sample data: {df_sample.shape} records for {n_products} products")
        return df_sample
    
    def get_complete_dataset(self) -> pd.DataFrame:
        """Get the complete preprocessed dataset"""
        return self.preprocess_data()
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'processed_superstore_data.csv'):
        """Save processed dataset to CSV file"""
        df.to_csv(filename, index=False)
        print(f"✅ Processed dataset saved to {filename}")
        return filename


# Example usage
if __name__ == "__main__":
    loader = SuperstoreDataLoader()
    
    # Load and preprocess data
    df = loader.get_complete_dataset()
    
    # Basic statistics
    print("\n📊 Dataset Statistics:")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Products: {df['product_id'].nunique()}")
    print(f"Categories: {df['category'].unique()}")
    print(f"Total revenue: ${df['revenue'].sum():,.2f}")
    print(f"Total quantity sold: {df['quantity_sold'].sum():,}")
    
    # Save processed data
    loader.save_processed_data(df) 