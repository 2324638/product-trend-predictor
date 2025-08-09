#!/bin/bash

# Azure App Service startup script for FastAPI application
echo "🚀 Starting AI Product Trend Predictor..."
echo "📅 Timestamp: $(date)"
echo "📁 Current directory: $(pwd)"
echo "📂 Directory contents:"
ls -la

# Set working directory
cd /home/site/wwwroot || {
    echo "❌ Failed to change to /home/site/wwwroot"
    echo "📂 Available directories:"
    ls -la /home/site/
    # Try current directory as fallback
    echo "🔄 Using current directory as fallback"
}

# Display current directory and contents
echo "📁 Working directory: $(pwd)"
echo "📂 Contents:"
ls -la

# Ensure Python path is set
export PYTHONPATH=/home/site/wwwroot:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models/saved
mkdir -p dashboard/static
mkdir -p logs
mkdir -p evaluation_results

# Check if main.py exists, if not try app.py
if [ -f "main.py" ]; then
    ENTRY_POINT="main.py"
    echo "✅ Found main.py"
elif [ -f "app.py" ]; then
    ENTRY_POINT="app.py"
    echo "✅ Found app.py as fallback"
else
    echo "❌ ERROR: Neither main.py nor app.py found"
    echo "📂 Available Python files:"
    find . -name "*.py" -type f | head -10
    exit 1
fi

# Check if requirements file exists
if [ -f "requirements-azure.txt" ]; then
    REQ_FILE="requirements-azure.txt"
elif [ -f "requirements.txt" ]; then
    REQ_FILE="requirements.txt"
else
    echo "⚠️  No requirements file found, skipping dependency installation"
    REQ_FILE=""
fi

# Install dependencies if requirements file exists
if [ ! -z "$REQ_FILE" ]; then
    echo "📦 Installing dependencies from $REQ_FILE..."
    python -m pip install --upgrade pip
    python -m pip install -r "$REQ_FILE"
fi

# Start the application
echo "🌐 Starting FastAPI application..."
echo "🔧 Entry point: $ENTRY_POINT"
echo "🚀 Port: ${PORT:-8000}"

python "$ENTRY_POINT" 