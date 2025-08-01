#!/usr/bin/env python3
"""
Simplified AI Product Trend Prediction Demo
Runs with minimal dependencies for testing
"""

import json
import time
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import os

class TrendPredictionHandler(SimpleHTTPRequestHandler):
    """Simple HTTP handler for the trend prediction demo"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.serve_dashboard()
        elif self.path == '/api/health':
            self.serve_json({"status": "healthy", "demo_mode": True})
        elif self.path == '/api/products':
            self.serve_sample_products()
        elif self.path.startswith('/api/predict'):
            self.serve_sample_prediction()
        else:
            super().do_GET()
    
    def serve_dashboard(self):
        """Serve a simple HTML dashboard"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Trend Predictor - Demo Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .demo-banner { background: #f39c12; color: white; padding: 15px; text-align: center; border-radius: 5px; margin-bottom: 20px; }
        .section { margin: 20px 0; padding: 20px; background: #ecf0f1; border-radius: 5px; }
        .button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .button:hover { background: #2980b9; }
        .prediction-result { background: #27ae60; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .instructions { background: #e8f4fd; padding: 20px; border-left: 4px solid #3498db; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AI Product Trend Predictor</h1>
            <p>Advanced E-commerce Trend Prediction System</p>
        </div>
        
        <div class="demo-banner">
            📱 DEMO MODE - Simplified version running without full dependencies
        </div>
        
        <div class="section">
            <h2>🎯 Features Available in Full Version:</h2>
            <ul>
                <li>✅ Multi-Model AI Ensemble (LSTM + XGBoost + LightGBM)</li>
                <li>✅ Real-time Trend Prediction</li>
                <li>✅ Interactive Web Dashboard</li>
                <li>✅ REST API for Integration</li>
                <li>✅ Advanced Feature Engineering</li>
                <li>✅ Automated Alerts System</li>
                <li>✅ Data Upload & Management</li>
                <li>✅ Model Performance Monitoring</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>🔮 Demo Prediction</h2>
            <p>Select a product and generate a sample prediction:</p>
            <select id="product-select">
                <option value="PROD_0001">Electronics Product 1</option>
                <option value="PROD_0002">Clothing Product 2</option>
                <option value="PROD_0003">Home & Garden Product 3</option>
            </select>
            <input type="number" id="days-ahead" value="30" min="1" max="365" placeholder="Days ahead">
            <button class="button" onclick="generatePrediction()">Generate Prediction</button>
            
            <div id="prediction-output"></div>
        </div>
        
        <div class="instructions">
            <h2>📋 To Run Full Version:</h2>
            <ol>
                <li><strong>Install Dependencies:</strong>
                    <pre>pip install pandas numpy scikit-learn fastapi uvicorn tensorflow xgboost lightgbm</pre>
                </li>
                <li><strong>Run Application:</strong>
                    <pre>python main.py</pre>
                </li>
                <li><strong>Access Dashboard:</strong>
                    <pre>http://localhost:8000</pre>
                </li>
            </ol>
        </div>
        
        <div class="section">
            <h2>🔗 API Endpoints (Demo)</h2>
            <ul>
                <li><a href="/api/health">Health Check</a></li>
                <li><a href="/api/products">Sample Products</a></li>
                <li><a href="/api/predict?product_id=PROD_0001&days_ahead=30">Sample Prediction</a></li>
            </ul>
        </div>
    </div>
    
    <script>
        function generatePrediction() {
            const productId = document.getElementById('product-select').value;
            const daysAhead = document.getElementById('days-ahead').value;
            const output = document.getElementById('prediction-output');
            
            // Simulate API call
            output.innerHTML = '<p>⏳ Generating prediction...</p>';
            
            setTimeout(() => {
                const trendDirection = Math.random() > 0.5 ? 'up' : 'down';
                const trendStrength = (Math.random() * 100).toFixed(1);
                const accuracy = (85 + Math.random() * 10).toFixed(1);
                
                output.innerHTML = `
                    <div class="prediction-result">
                        <h3>📊 Prediction Results for ${productId}</h3>
                        <p><strong>Trend Direction:</strong> ${trendDirection.toUpperCase()}</p>
                        <p><strong>Trend Strength:</strong> ${trendStrength}%</p>
                        <p><strong>Model Accuracy:</strong> ${accuracy}%</p>
                        <p><strong>Prediction Period:</strong> ${daysAhead} days</p>
                        <p><em>Note: This is a demo prediction. Install full version for real AI predictions.</em></p>
                    </div>
                `;
            }, 2000);
        }
    </script>
</body>
</html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_json(self, data):
        """Serve JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def serve_sample_products(self):
        """Serve sample products data"""
        products = {
            "total_products": 3,
            "products": [
                {"product_id": "PROD_0001", "name": "Electronics Product 1", "category": "Electronics", "total_sales": 1500},
                {"product_id": "PROD_0002", "name": "Clothing Product 2", "category": "Clothing", "total_sales": 890},
                {"product_id": "PROD_0003", "name": "Home & Garden Product 3", "category": "Home & Garden", "total_sales": 654}
            ]
        }
        self.serve_json(products)
    
    def serve_sample_prediction(self):
        """Serve sample prediction data"""
        import random
        prediction = {
            "product_id": "PROD_0001",
            "trend_direction": random.choice(["up", "down", "stable"]),
            "trend_strength": round(random.random(), 2),
            "model_accuracy": round(0.85 + random.random() * 0.1, 2),
            "predictions": [
                {"date": (datetime.now() + timedelta(days=i)).isoformat(), "predicted_quantity": random.randint(10, 50)}
                for i in range(1, 31)
            ]
        }
        self.serve_json(prediction)

def main():
    """Run the simple demo server"""
    print("=" * 60)
    print("🚀 AI Product Trend Prediction - Demo Mode")
    print("=" * 60)
    print("Starting simple HTTP server...")
    print("This is a demo version with minimal dependencies.")
    print("For full functionality, install all requirements.")
    print()
    
    port = 8080
    server = HTTPServer(('localhost', port), TrendPredictionHandler)
    
    print(f"✅ Demo server running at: http://localhost:{port}")
    print("🔄 Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo server stopped")
        print("👋 To run the full version, install dependencies and use: python main.py")

if __name__ == "__main__":
    main()