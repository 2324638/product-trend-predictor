#!/bin/bash

# Azure App Service startup script for FastAPI application
echo "🚀 Starting AI Product Trend Predictor on Azure..."
echo "📅 Timestamp: $(date)"
echo "📁 Current directory: $(pwd)"
echo "📂 Directory contents:"
ls -la

# Azure App Service specific paths
AZURE_WWWROOT="/home/site/wwwroot"
CURRENT_DIR=$(pwd)

# Try to find the correct working directory
if [ -d "$AZURE_WWWROOT" ]; then
    echo "✅ Found Azure wwwroot directory: $AZURE_WWWROOT"
    cd "$AZURE_WWWROOT"
elif [ -d "/home/site/wwwroot" ]; then
    echo "✅ Found Azure wwwroot directory: /home/site/wwwroot"
    cd "/home/site/wwwroot"
elif [ -d "/tmp/zipdeploy/extracted" ]; then
    echo "✅ Found extracted deployment directory: /tmp/zipdeploy/extracted"
    cd "/tmp/zipdeploy/extracted"
else
    echo "⚠️  Using current directory as fallback: $CURRENT_DIR"
fi

# Display current directory and contents
echo "📁 Working directory: $(pwd)"
echo "📂 Contents:"
ls -la

# Ensure Python path is set
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models/saved
mkdir -p dashboard/static
mkdir -p logs
mkdir -p evaluation_results

# List all Python files to debug
echo "🔍 Searching for Python files..."
find . -name "*.py" -type f | head -20

# Check if main.py exists, if not try app.py
if [ -f "main.py" ]; then
    ENTRY_POINT="main.py"
    echo "✅ Found main.py"
elif [ -f "app.py" ]; then
    ENTRY_POINT="app.py"
    echo "✅ Found app.py as fallback"
else
    echo "❌ ERROR: Neither main.py nor app.py found"
    echo "📂 Available Python files in current directory:"
    ls -la *.py 2>/dev/null || echo "No .py files in current directory"
    echo "📂 Available Python files recursively:"
    find . -name "*.py" -type f | head -10
    echo "📂 Directory structure:"
    find . -type d -maxdepth 3 | head -15
    exit 1
fi

# Check if requirements file exists
if [ -f "requirements-azure.txt" ]; then
    REQ_FILE="requirements-azure.txt"
    echo "✅ Found requirements-azure.txt"
elif [ -f "requirements.txt" ]; then
    REQ_FILE="requirements.txt"
    echo "✅ Found requirements.txt"
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
echo "🐍 Python path: $PYTHONPATH"

# Use gunicorn for production deployment
if command -v gunicorn &> /dev/null; then
    echo "🚀 Using gunicorn for production deployment..."
    gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 4 --timeout 120 "$ENTRY_POINT:app"
else
    echo "🚀 Using uvicorn for development deployment..."
    python "$ENTRY_POINT"
fi 