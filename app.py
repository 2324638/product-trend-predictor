#!/usr/bin/env python3
"""
Azure App Service entry point for AI Product Trend Predictor
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ.setdefault('PYTHONPATH', str(project_root))
os.environ.setdefault('PYTHONUNBUFFERED', '1')

def create_app():
    """Create and configure the FastAPI application"""
    try:
        # Import the FastAPI app
        from api.main import app
        
        # Create necessary directories
        directories = ['models/saved', 'dashboard/static', 'logs', 'evaluation_results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"✅ Application created successfully")
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"🐍 Python path: {sys.path[:3]}")
        
        return app
        
    except Exception as e:
        print(f"❌ Failed to create application: {e}")
        import traceback
        traceback.print_exc()
        raise

# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Product Trend Predictor for Azure...")
    print("🌐 Server will be available at: http://localhost:8000")
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
        access_log=True
    ) 