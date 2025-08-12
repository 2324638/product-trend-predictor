@echo off
echo 🚀 Starting AI Product Trend Predictor...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
if not exist "models\saved" (
    echo 📦 Installing dependencies...
    python -m pip install -r requirements.txt
)

REM Start the application
echo 🌐 Starting the application...
echo 📍 Dashboard will be available at: http://localhost:8000
echo 📍 API Docs will be available at: http://localhost:8000/docs
echo.
echo 💡 Quick Start:
echo    1. Open http://localhost:8000 in your browser
echo    2. Go to 'Data Management' tab
echo    3. Click 'Load Superstore Data' to load your dataset
echo    4. Click 'Train Model' to train the AI models
echo    5. Go to 'Predictions' tab to make predictions
echo.
echo 🔄 Press Ctrl+C to stop the server
echo.

python main.py
pause 