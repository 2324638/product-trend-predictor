import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random
from data.models import ProductData, SalesData, MarketData, SeasonalData


class EcommerceDataGenerator:
    """
    Generate synthetic e-commerce data for trend prediction model training and testing
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Product categories and their characteristics
        self.categories = {
            'Electronics': {'base_price': 200, 'price_std': 100, 'seasonality': 1.2},
            'Clothing': {'base_price': 50, 'price_std': 30, 'seasonality': 1.5},
            'Home & Garden': {'base_price': 80, 'price_std': 40, 'seasonality': 1.1},
            'Sports': {'base_price': 70, 'price_std': 35, 'seasonality': 1.3},
            'Books': {'base_price': 20, 'price_std': 10, 'seasonality': 0.9},
            'Beauty': {'base_price': 40, 'price_std': 20, 'seasonality': 1.1},
            'Toys': {'base_price': 30, 'price_std': 15, 'seasonality': 2.0},
            'Food': {'base_price': 25, 'price_std': 15, 'seasonality': 0.8}
        }
        
        self.brands = ['BrandA']
        self.channels = ['online', 'store', 'mobile']
        
    def generate_products(self, n_products: int = 100) -> List[ProductData]:
        """Generate synthetic product data"""
        products = []
        
        for i in range(n_products):
            category = random.choice(list(self.categories.keys()))
            category_info = self.categories[category]
            
            price = max(5, np.random.normal(
                category_info['base_price'], 
                category_info['price_std']
            ))
            
            product = ProductData(
                product_id=f"PROD_{i:04d}",
                name=f"{category} Product {i}",
                category=category,
                brand=random.choice(self.brands),
                price=round(price, 2)
            )
            products.append(product)
            
        return products
    
    def generate_seasonal_factors(self, date: datetime, category: str) -> float:
        """Generate seasonal multipliers based on date and category"""
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        # Base seasonal pattern
        seasonal_base = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Category-specific patterns
        category_multiplier = self.categories[category]['seasonality']
        
        # Holiday effects
        holiday_boost = 1.0
        if month == 12:  # December holiday boost
            holiday_boost = 1.8
        elif month == 11:  # November (Black Friday)
            holiday_boost = 1.5
        elif month in [6, 7]:  # Summer boost for certain categories
            if category in ['Sports', 'Clothing']:
                holiday_boost = 1.3
        elif month in [1, 2]:  # Post-holiday slump
            holiday_boost = 0.7
            
        return seasonal_base * category_multiplier * holiday_boost
    
    def generate_trend_factor(self, date: datetime, product_id: str, 
                            trend_type: str = 'random') -> float:
        """Generate trend factors for products"""
        days_since_epoch = (date - datetime(2020, 1, 1)).days
        
        if trend_type == 'growing':
            return 1 + 0.001 * days_since_epoch
        elif trend_type == 'declining':
            return max(0.1, 1 - 0.0005 * days_since_epoch)
        elif trend_type == 'cyclical':
            return 1 + 0.2 * np.sin(2 * np.pi * days_since_epoch / 90)
        else:  # random walk
            # Use product_id as seed for consistency
            np.random.seed(hash(product_id) % 2**32)
            trend = 1.0
            for _ in range(days_since_epoch):
                trend *= np.random.normal(1.0, 0.005)
            trend = max(0.1, min(3.0, trend))  # Bound the trend
            np.random.seed()  # Reset seed
            return trend
    
    def generate_sales_data(self, products: List[ProductData], 
                          start_date: datetime, end_date: datetime) -> List[SalesData]:
        """Generate synthetic sales data"""
        sales_data = []
        current_date = start_date
        
        while current_date <= end_date:
            for product in products:
                # Base demand influenced by multiple factors
                base_demand = np.random.poisson(10)
                
                # Seasonal effects
                seasonal_factor = self.generate_seasonal_factors(current_date, product.category)
                
                # Trend effects
                trend_factor = self.generate_trend_factor(current_date, product.product_id)
                
                # Weekend effects
                weekend_factor = 1.3 if current_date.weekday() >= 5 else 1.0
                
                # Price elasticity
                price_factor = max(0.1, 2.0 - (product.price / 100))
                
                # Random noise
                noise_factor = np.random.normal(1.0, 0.2)
                
                # Calculate final quantity
                quantity = max(0, int(base_demand * seasonal_factor * trend_factor * 
                                    weekend_factor * price_factor * noise_factor))
                
                if quantity > 0:  # Only record sales if quantity > 0
                    # Generate discount
                    discount = 0
                    if random.random() < 0.2:  # 20% chance of discount
                        discount = random.uniform(5, 30)
                    
                    # Channel selection
                    channel = random.choice(self.channels)
                    
                    # Revenue calculation
                    effective_price = product.price * (1 - discount / 100)
                    revenue = quantity * effective_price
                    
                    sale = SalesData(
                        product_id=product.product_id,
                        date=current_date,
                        quantity_sold=quantity,
                        revenue=round(revenue, 2),
                        customer_count=max(1, int(quantity * random.uniform(0.7, 1.0))),
                        discount_applied=discount,
                        channel=channel
                    )
                    sales_data.append(sale)
            
            current_date += timedelta(days=1)
        
        return sales_data
    
    def generate_market_data(self, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Generate synthetic market data"""
        market_data = []
        current_date = start_date
        
        # Initialize market conditions
        market_index = 100.0
        
        while current_date <= end_date:
            # Market index random walk
            market_change = np.random.normal(0.001, 0.02)
            market_index *= (1 + market_change)
            market_index = max(50, min(200, market_index))  # Bound the index
            
            # Generate other market factors
            competitor_price_avg = np.random.normal(50, 15)
            search_volume = max(0, int(np.random.normal(1000, 300)))
            social_mentions = max(0, int(np.random.poisson(50)))
            economic_indicator = np.random.normal(0.02, 0.01)  # Economic growth rate
            
            market = MarketData(
                date=current_date,
                market_index=round(market_index, 2),
                competitor_price_avg=round(competitor_price_avg, 2),
                search_volume=search_volume,
                social_mentions=social_mentions,
                economic_indicator=round(economic_indicator, 4)
            )
            market_data.append(market)
            
            current_date += timedelta(days=1)
            
        return market_data
    
    def generate_seasonal_data(self, start_date: datetime, end_date: datetime) -> List[SeasonalData]:
        """Generate seasonal and temporal data"""
        seasonal_data = []
        current_date = start_date
        
        # US federal holidays (simplified)
        holidays = [
            (1, 1),   # New Year's Day
            (2, 14),  # Valentine's Day
            (3, 17),  # St. Patrick's Day
            (7, 4),   # Independence Day
            (10, 31), # Halloween
            (11, 25), # Thanksgiving (approximation)
            (12, 25), # Christmas
        ]
        
        while current_date <= end_date:
            # Check if it's a holiday
            is_holiday = (current_date.month, current_date.day) in holidays
            
            # Determine season
            month = current_date.month
            if month in [12, 1, 2]:
                season = 'winter'
            elif month in [3, 4, 5]:
                season = 'spring'
            elif month in [6, 7, 8]:
                season = 'summer'
            else:
                season = 'fall'
            
            seasonal = SeasonalData(
                date=current_date,
                is_weekend=current_date.weekday() >= 5,
                is_holiday=is_holiday,
                month=current_date.month,
                quarter=(current_date.month - 1) // 3 + 1,
                day_of_year=current_date.timetuple().tm_yday,
                week_of_year=current_date.isocalendar()[1],
                season=season
            )
            seasonal_data.append(seasonal)
            
            current_date += timedelta(days=1)
            
        return seasonal_data
    
    def generate_complete_dataset(self, n_products: int = 50, 
                                 start_date: datetime = None,
                                 end_date: datetime = None) -> pd.DataFrame:
        """Generate complete dataset with all components"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)  # 2 years of data
        if end_date is None:
            end_date = datetime.now() - timedelta(days=1)
            
        print(f"Generating data for {n_products} products from {start_date.date()} to {end_date.date()}")
        
        # Generate all components
        products = self.generate_products(n_products)
        sales_data = self.generate_sales_data(products, start_date, end_date)
        market_data = self.generate_market_data(start_date, end_date)
        seasonal_data = self.generate_seasonal_data(start_date, end_date)
        
        # Convert to DataFrames
        products_df = pd.DataFrame([p.dict() for p in products])
        sales_df = pd.DataFrame([s.dict() for s in sales_data])
        market_df = pd.DataFrame([m.dict() for m in market_data])
        seasonal_df = pd.DataFrame([s.dict() for s in seasonal_data])
        
        # Merge all data
        combined_df = sales_df.merge(products_df, on='product_id', how='left')
        combined_df = combined_df.merge(market_df, on='date', how='left')
        combined_df = combined_df.merge(seasonal_df, on='date', how='left')
        
        # Add some noise and missing values to make it more realistic
        # Randomly set some market data to missing
        mask = np.random.random(len(combined_df)) < 0.05
        combined_df.loc[mask, 'competitor_price_avg'] = np.nan
        
        mask = np.random.random(len(combined_df)) < 0.03
        combined_df.loc[mask, 'search_volume'] = np.nan
        
        print(f"Generated {len(combined_df)} sales records")
        print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"Products: {combined_df['product_id'].nunique()}")
        print(f"Categories: {combined_df['category'].nunique()}")
        
        return combined_df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'ecommerce_data.csv'):
        """Save dataset to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        return filename


# Example usage and testing
if __name__ == "__main__":
    generator = EcommerceDataGenerator(seed=42)
    
    # Generate dataset
    df = generator.generate_complete_dataset(
        n_products=30,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nSample data:")
    print(df.head())
    
    # Save dataset
    generator.save_dataset(df)