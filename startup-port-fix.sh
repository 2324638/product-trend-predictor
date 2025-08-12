#!/bin/bash

# Azure App Service Startup Script - Port Conflict Fix
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

# Azure App Service Port Configuration and Conflict Resolution
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

echo "🚀 Target port: $APP_PORT"

# Port conflict resolution
echo "🔧 Resolving port conflicts..."

# Check what's using the target port
if command -v netstat &> /dev/null; then
    echo "🔍 Checking what's using port $APP_PORT..."
    netstat -tlnp | grep ":$APP_PORT " || echo "Port $APP_PORT appears to be available"
fi

# Try to kill any processes using the target port
if command -v fuser &> /dev/null; then
    echo "🔧 Attempting to free port $APP_PORT..."
    fuser -k $APP_PORT/tcp 2>/dev/null || echo "No processes to kill on port $APP_PORT"
fi

# Alternative: try different ports if the main one is busy
PORT_ATTEMPTS=("$APP_PORT" "8000" "8001" "8002" "8003")
FINAL_PORT=""

for port in "${PORT_ATTEMPTS[@]}"; do
    echo "🔍 Testing port $port..."
    if command -v netstat &> /dev/null; then
        if ! netstat -tlnp | grep -q ":$port "; then
            echo "✅ Port $port is available"
            FINAL_PORT="$port"
            break
        else
            echo "❌ Port $port is in use"
        fi
    else
        # If netstat not available, just use the first port
        FINAL_PORT="$port"
        break
    fi
done

if [ -z "$FINAL_PORT" ]; then
    echo "⚠️  All ports are busy, using target port anyway"
    FINAL_PORT="$APP_PORT"
fi

echo "🚀 Final port selection: $FINAL_PORT"

# Set the port environment variable for the Python app
export PORT="$FINAL_PORT"
export WEBSITES_PORT="$FINAL_PORT"

# Start the application with simple Python execution
echo "🌐 Starting FastAPI application..."
echo "🔧 Entry point: $ENTRY_POINT"
echo "🐍 Python path: $PYTHONPATH"
echo "🚀 Using port: $FINAL_PORT"

# Simple approach: just run the Python file directly
echo "🚀 Starting with Python directly..."
echo "🔧 Running: python $ENTRY_POINT with PORT=$FINAL_PORT"

# Run the Python application directly
exec python "$ENTRY_POINT" 