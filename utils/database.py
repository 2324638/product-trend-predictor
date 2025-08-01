import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import os


class DatabaseManager:
    """
    Simple database manager for storing e-commerce data and predictions
    Uses SQLite for simplicity
    """
    
    def __init__(self, db_path: str = "ecommerce_trends.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Products table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    brand TEXT,
                    price REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Sales data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sales_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT,
                    date DATE,
                    quantity_sold INTEGER,
                    revenue REAL,
                    customer_count INTEGER DEFAULT 1,
                    discount_applied REAL DEFAULT 0.0,
                    channel TEXT DEFAULT 'online',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products (product_id)
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT,
                    prediction_date DATE,
                    predicted_quantity REAL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    trend_direction TEXT,
                    trend_strength REAL,
                    model_accuracy REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products (product_id)
                )
            ''')
            
            # Model metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    mae REAL,
                    mse REAL,
                    rmse REAL,
                    mape REAL,
                    r2_score REAL,
                    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    confidence REAL,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (product_id) REFERENCES products (product_id)
                )
            ''')
            
            conn.commit()
    
    def insert_products(self, products_df: pd.DataFrame):
        """Insert products data"""
        with sqlite3.connect(self.db_path) as conn:
            products_df.to_sql('products', conn, if_exists='replace', index=False)
    
    def insert_sales_data(self, sales_df: pd.DataFrame):
        """Insert sales data"""
        with sqlite3.connect(self.db_path) as conn:
            # Ensure date column is in correct format
            sales_df['date'] = pd.to_datetime(sales_df['date']).dt.date
            sales_df.to_sql('sales_data', conn, if_exists='append', index=False)
    
    def insert_prediction(self, prediction_data: Dict[str, Any]):
        """Insert a single prediction"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (product_id, prediction_date, predicted_quantity, confidence_lower, 
                 confidence_upper, trend_direction, trend_strength, model_accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data['product_id'],
                prediction_data['prediction_date'],
                prediction_data['predicted_quantity'],
                prediction_data.get('confidence_lower'),
                prediction_data.get('confidence_upper'),
                prediction_data['trend_direction'],
                prediction_data['trend_strength'],
                prediction_data['model_accuracy']
            ))
            
            conn.commit()
    
    def insert_model_metrics(self, metrics: List[Dict[str, Any]]):
        """Insert model performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for metric in metrics:
                cursor.execute('''
                    INSERT INTO model_metrics 
                    (model_name, mae, mse, rmse, mape, r2_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric['model_name'],
                    metric['mae'],
                    metric['mse'],
                    metric['rmse'],
                    metric['mape'],
                    metric['r2_score']
                ))
            
            conn.commit()
    
    def insert_alert(self, alert_data: Dict[str, Any]):
        """Insert an alert"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts 
                (product_id, alert_type, severity, message, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                alert_data['product_id'],
                alert_data['alert_type'],
                alert_data['severity'],
                alert_data['message'],
                alert_data['confidence']
            ))
            
            conn.commit()
    
    def get_products(self) -> pd.DataFrame:
        """Get all products"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query('SELECT * FROM products', conn)
    
    def get_sales_data(self, product_id: Optional[str] = None, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """Get sales data with optional filters"""
        query = 'SELECT * FROM sales_data WHERE 1=1'
        params = []
        
        if product_id:
            query += ' AND product_id = ?'
            params.append(product_id)
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)
        
        query += ' ORDER BY date'
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_predictions(self, product_id: Optional[str] = None) -> pd.DataFrame:
        """Get predictions with optional product filter"""
        query = 'SELECT * FROM predictions WHERE 1=1'
        params = []
        
        if product_id:
            query += ' AND product_id = ?'
            params.append(product_id)
        
        query += ' ORDER BY prediction_date'
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_latest_model_metrics(self) -> pd.DataFrame:
        """Get latest model metrics"""
        query = '''
            SELECT model_name, mae, mse, rmse, mape, r2_score, training_date
            FROM model_metrics 
            WHERE training_date = (
                SELECT MAX(training_date) FROM model_metrics
            )
        '''
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def get_alerts(self, acknowledged: bool = False) -> pd.DataFrame:
        """Get alerts"""
        query = 'SELECT * FROM alerts WHERE acknowledged = ? ORDER BY detected_at DESC'
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=[acknowledged])
    
    def acknowledge_alert(self, alert_id: int):
        """Mark an alert as acknowledged"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE alerts SET acknowledged = TRUE WHERE id = ?', (alert_id,))
            conn.commit()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count records in each table
            tables = ['products', 'sales_data', 'predictions', 'model_metrics', 'alerts']
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Get date range for sales data
            cursor.execute('SELECT MIN(date), MAX(date) FROM sales_data')
            date_range = cursor.fetchone()
            if date_range[0] and date_range[1]:
                stats['sales_date_range'] = {
                    'start': date_range[0],
                    'end': date_range[1]
                }
            
            # Get file size
            if os.path.exists(self.db_path):
                stats['database_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
            
            return stats
    
    def backup_database(self, backup_path: Optional[str] = None):
        """Create a backup of the database"""
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"backup_ecommerce_trends_{timestamp}.db"
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        return backup_path
    
    def export_to_csv(self, table_name: str, output_path: str):
        """Export a table to CSV"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
            df.to_csv(output_path, index=False)
            return output_path
    
    def clear_old_predictions(self, days_old: int = 30):
        """Clear predictions older than specified days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM predictions 
                WHERE created_at < datetime('now', '-{} days')
            '''.format(days_old))
            conn.commit()
            return cursor.rowcount


# Example usage and testing
if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager("test_ecommerce.db")
    
    # Create sample data
    products_data = pd.DataFrame({
        'product_id': ['PROD_001', 'PROD_002'],
        'name': ['Test Product 1', 'Test Product 2'],
        'category': ['Electronics', 'Clothing'],
        'brand': ['BrandA', 'BrandB'],
        'price': [99.99, 49.99]
    })
    
    sales_data = pd.DataFrame({
        'product_id': ['PROD_001', 'PROD_002'],
        'date': ['2023-01-01', '2023-01-01'],
        'quantity_sold': [10, 5],
        'revenue': [999.90, 249.95],
        'customer_count': [10, 5],
        'discount_applied': [0.0, 10.0],
        'channel': ['online', 'store']
    })
    
    # Insert data
    db.insert_products(products_data)
    db.insert_sales_data(sales_data)
    
    # Insert sample prediction
    prediction = {
        'product_id': 'PROD_001',
        'prediction_date': '2023-01-02',
        'predicted_quantity': 12.5,
        'confidence_lower': 8.0,
        'confidence_upper': 17.0,
        'trend_direction': 'up',
        'trend_strength': 0.75,
        'model_accuracy': 0.85
    }
    db.insert_prediction(prediction)
    
    # Get statistics
    stats = db.get_database_stats()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTest completed successfully!")