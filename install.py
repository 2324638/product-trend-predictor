#!/usr/bin/env python3
"""
Installation and setup script for AI Product Trend Prediction
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    """Print installation header"""
    print("=" * 70)
    print("🚀 AI-Powered Product Trend Prediction Installation")
    print("=" * 70)
    print()


def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "models/saved",
        "dashboard/static",
        "data",
        "utils",
        "logs",
        "evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")


def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        # Update pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("   Please install manually using: pip install -r requirements.txt")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        print("   Please ensure you're running this script from the project root")
        return False


def check_optional_dependencies():
    """Check for optional dependencies"""
    print("\n🔧 Checking optional dependencies...")
    
    optional_packages = {
        "tensorflow": "TensorFlow (for LSTM models)",
        "xgboost": "XGBoost (for gradient boosting)",
        "lightgbm": "LightGBM (for fast gradient boosting)",
        "plotly": "Plotly (for interactive charts)",
        "fastapi": "FastAPI (for web API)",
        "uvicorn": "Uvicorn (for web server)"
    }
    
    missing_packages = []
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ❌ {description} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
    
    return len(missing_packages) == 0


def generate_sample_data():
    """Generate initial sample data"""
    print("\n🎲 Generating sample data...")
    
    try:
        from data.data_generator import EcommerceDataGenerator
        from datetime import datetime, timedelta
        
        generator = EcommerceDataGenerator(seed=42)
        df = generator.generate_complete_dataset(
            n_products=20,
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now() - timedelta(days=1)
        )
        
        # Save sample data
        df.to_csv("sample_ecommerce_data.csv", index=False)
        print("✅ Sample data generated and saved to sample_ecommerce_data.csv")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Could not generate sample data: {e}")
        print("   You can generate data later from the web interface")
        return False


def run_health_check():
    """Run a basic health check"""
    print("\n🏥 Running health check...")
    
    try:
        # Test data models
        from data.models import ProductData, SalesData
        print("   ✅ Data models working")
        
        # Test preprocessor
        from data.preprocessor import TrendDataPreprocessor
        preprocessor = TrendDataPreprocessor()
        print("   ✅ Data preprocessor working")
        
        # Test database
        from utils.database import DatabaseManager
        db = DatabaseManager(":memory:")  # In-memory database for testing
        print("   ✅ Database manager working")
        
        print("✅ All components healthy")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def create_startup_script():
    """Create platform-specific startup scripts"""
    print("\n📝 Creating startup scripts...")
    
    # Windows batch file
    if platform.system() == "Windows":
        with open("start.bat", "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting AI Trend Predictor...\n")
            f.write("python main.py\n")
            f.write("pause\n")
        print("   ✅ start.bat created for Windows")
    
    # Unix shell script
    with open("start.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Starting AI Trend Predictor...'\n")
        f.write("python3 main.py\n")
    
    # Make shell script executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("start.sh", 0o755)
        print("   ✅ start.sh created for Unix/Linux/MacOS")


def print_completion_message():
    """Print installation completion message"""
    print("\n" + "=" * 70)
    print("🎉 Installation Complete!")
    print("=" * 70)
    print()
    print("📋 What's installed:")
    print("   • AI trend prediction models (LSTM, XGBoost, LightGBM)")
    print("   • Interactive web dashboard")
    print("   • REST API for integration")
    print("   • Data management tools")
    print("   • Model evaluation utilities")
    print()
    print("🚀 Quick Start:")
    print("   1. Run: python main.py")
    print("   2. Open: http://localhost:8000")
    print("   3. Generate sample data in 'Data Management' tab")
    print("   4. Train models")
    print("   5. Make predictions!")
    print()
    print("📚 Documentation:")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • README.md: Project overview and usage")
    print()
    
    if platform.system() == "Windows":
        print("💡 You can also run: start.bat")
    else:
        print("💡 You can also run: ./start.sh")
    
    print()
    print("🆘 Need help? Check the README.md file or API documentation")
    print("=" * 70)


def main():
    """Main installation function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Installation completed with warnings")
        print("   Please install dependencies manually and run again")
    
    # Check optional dependencies
    all_deps_ok = check_optional_dependencies()
    
    # Generate sample data
    generate_sample_data()
    
    # Run health check
    health_ok = run_health_check()
    
    # Create startup scripts
    create_startup_script()
    
    # Print completion message
    print_completion_message()
    
    # Exit with appropriate code
    if all_deps_ok and health_ok:
        print("✅ Installation successful! Ready to use.")
        sys.exit(0)
    else:
        print("⚠️  Installation completed with warnings.")
        print("   The system should work but some features may be limited.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        print("   Please check the error and try again")
        sys.exit(1)