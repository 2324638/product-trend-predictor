#!/bin/bash

# Azure App Service Startup Script - Fixed for Port Conflicts
echo "🚀 Starting AI Product Trend Predictor on Azure..."
echo "📅 Timestamp: $(date)"
echo "📁 Current directory: $(pwd)"

# Azure App Service specific paths - Oryx build system
AZURE_REPOSITORY="/home/site/repository"
AZURE_WWWROOT="/home/site/wwwroot"
CURRENT_DIR=$(pwd)

echo "🔍 Searching for Python files in all possible locations..."

# Check all possible Azure paths
for path in "$AZURE_REPOSITORY" "$AZURE_WWWROOT" "$CURRENT_DIR" "/tmp/zipdeploy/extracted"; do
    if [ -d "$path" ]; then
        echo "📂 Checking path: $path"
        if [ -f "$path/main.py" ]; then
            echo "✅ Found main.py in: $path"
            cd "$path"
            break
        elif [ -f "$path/app.py" ]; then
            echo "✅ Found app.py in: $path"
            cd "$path"
            break
        else
            echo "❌ No main.py or app.py found in: $path"
        fi
    fi
done

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

# Check if main.py exists, if not try app.py
if [ -f "main.py" ]; then
    ENTRY_POINT="main.py"
    echo "✅ Found main.py"
elif [ -f "app.py" ]; then
    ENTRY_POINT="app.py"
    echo "✅ Found app.py as fallback"
else
    echo "❌ ERROR: Neither main.py nor app.py found"
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

# Azure App Service Port Configuration
echo "🔍 Azure environment variables:"
echo "   PORT: ${PORT:-'not set'}"
echo "   WEBSITES_PORT: ${WEBSITES_PORT:-'not set'}"
echo "   HTTP_PLATFORM_PORT: ${HTTP_PLATFORM_PORT:-'not set'}"

# Determine the correct port to use
if [ ! -z "$PORT" ]; then
    APP_PORT="$PORT"
elif [ ! -z "$WEBSITES_PORT" ]; then
    APP_PORT="$WEBSITES_PORT"
elif [ ! -z "$HTTP_PLATFORM_PORT" ]; then
    APP_PORT="$HTTP_PLATFORM_PORT"
else
    APP_PORT="8000"
fi

echo "🚀 Using port: $APP_PORT"

# Start the application with proper Azure configuration
echo "🌐 Starting FastAPI application..."
echo "🔧 Entry point: $ENTRY_POINT"
echo "🐍 Python path: $PYTHONPATH"

# Use gunicorn for production deployment with Azure-specific settings
if command -v gunicorn &> /dev/null; then
    echo "🚀 Using gunicorn for Azure production deployment..."
    echo "🔧 Binding to: 0.0.0.0:$APP_PORT"
    
    # Azure App Service specific gunicorn configuration
    exec gunicorn \
        --bind 0.0.0.0:$APP_PORT \
        --workers 1 \
        --timeout 120 \
        --keep-alive 2 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --preload \
        --access-logfile - \
        --error-logfile - \
        --log-level info \
        "$ENTRY_POINT:app"
else
    echo "🚀 Using uvicorn for development deployment..."
    echo "🔧 Binding to: 0.0.0.0:$APP_PORT"
    
    # Use uvicorn with Azure-specific settings
    exec python -m uvicorn \
        "$ENTRY_POINT:app" \
        --host 0.0.0.0 \
        --port $APP_PORT \
        --log-level info \
        --access-log
fi 