#!/usr/bin/env python3
"""
AI-Powered Product Trend Prediction for E-commerce
Main application entry point
"""

import os
import sys
import uvicorn
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main application entry point"""
    print("=" * 60)
    print("🚀 AI-Powered Product Trend Prediction for E-commerce")
    print("=" * 60)
    print(f"⏰ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if required directories exist, create if not
    directories = ['models/saved', 'dashboard/static', 'data', 'utils']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    print()
    print("🔧 System Information:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Working Directory: {os.getcwd()}")
    print(f"   Project Root: {os.path.dirname(os.path.abspath(__file__))}")
    print()
    
    print("🎯 Features Available:")
    print("   ✅ Multi-Model AI Ensemble (LSTM + XGBoost + LightGBM + Prophet)")
    print("   ✅ Real-time Trend Prediction")
    print("   ✅ Interactive Web Dashboard")
    print("   ✅ REST API for Integration")
    print("   ✅ Advanced Feature Engineering")
    print("   ✅ Automated Alerts System")
    print("   ✅ Data Upload & Management")
    print("   ✅ Model Performance Monitoring")
    print("   ✅ Prophet Time Series Forecasting")
    print()
    
    # Import and start the FastAPI application
    try:
        from api.main import app
        
        print("🌐 Starting Web Server...")
        print("   Dashboard: http://localhost:8000")
        print("   API Docs:  http://localhost:8000/docs")
        print("   API Health: http://localhost:8000/health")
        print()
        print("💡 Quick Start:")
        print("   1. Open http://localhost:8000 in your browser")
        print("   2. Go to 'Data Management' tab")
        print("   3. Click 'Load Superstore Data' to load your dataset")
        print("   4. Click 'Train Model' to train the AI models")
        print("   5. Go to 'Predictions' tab to make predictions")
        print()
        print("🔄 Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import application: {e}")
        print("📦 Please install required dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("👋 Thank you for using AI Trend Predictor!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()