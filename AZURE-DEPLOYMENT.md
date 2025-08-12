# 🚀 Azure App Service Deployment Guide

This guide will help you deploy the AI Product Trend Predictor to Azure App Service and fix the deployment issues you're experiencing.

## ❌ Current Issue

The deployment is failing because:
1. **File Path Issues**: The startup script can't find `main.py` or `app.py`
2. **Line Ending Problems**: Windows CRLF line endings in shell scripts
3. **Azure Configuration**: Missing proper Azure App Service configuration

## ✅ Solution

I've created several files to fix these issues:

### 1. **startup-azure.sh** - Enhanced Azure Startup Script
- Better path detection for Azure environment
- Comprehensive debugging information
- Fallback mechanisms for different Azure paths
- Support for both gunicorn and uvicorn

### 2. **azure-app-service.json** - Azure Configuration
- Proper Python 3.11 runtime configuration
- Environment variables setup
- Startup command specification

### 3. **deploy-azure.ps1** - Automated Deployment Script
- PowerShell script for easy Azure deployment
- Resource group and App Service creation
- Proper configuration setup

## 🚀 Quick Fix Steps

### Option 1: Use the PowerShell Deployment Script (Recommended)

```powershell
# Navigate to your project directory
cd "C:\Users\2324638\OneDrive - Cognizant\Documents\product-trend-predictor"

# Run the deployment script
.\deploy-azure.ps1 -ResourceGroupName "ai-trend-predictor-rg" -AppServiceName "ai-trend-predictor-app"
```

### Option 2: Manual Azure Portal Deployment

1. **Create App Service**:
   - Go to Azure Portal → App Services → Create
   - Choose "Web App" with Python 3.11 runtime
   - Set startup command: `bash startup-azure.sh`

2. **Configure Environment Variables**:
   ```
   PYTHONPATH=/home/site/wwwroot
   PYTHONUNBUFFERED=1
   PORT=8000
   WEBSITES_PORT=8000
   SCM_DO_BUILD_DURING_DEPLOYMENT=true
   ```

3. **Deploy Your Code**:
   - Use VS Code Azure extension
   - Or zip and upload manually

### Option 3: Fix Current Deployment

If you want to fix your current deployment:

1. **Update the startup command** in Azure App Service Configuration:
   ```
   bash startup-azure.sh
   ```

2. **Set environment variables**:
   ```
   PYTHONPATH=/home/site/wwwroot
   SCM_DO_BUILD_DURING_DEPLOYMENT=true
   ```

3. **Redeploy** your application

## 🔧 Key Changes Made

### Fixed Line Endings
- `startup.sh` now has proper Unix line endings (LF)
- `startup-azure.sh` created with enhanced Azure support

### Enhanced Path Detection
- Script now searches multiple Azure paths
- Better debugging and error reporting
- Fallback mechanisms for different deployment scenarios

### Azure-Specific Configuration
- Proper Python runtime specification
- Environment variable configuration
- Startup command specification

## 📁 File Structure for Azure

```
product-trend-predictor/
├── startup-azure.sh          # Enhanced Azure startup script
├── azure-app-service.json    # Azure configuration
├── deploy-azure.ps1         # PowerShell deployment script
├── requirements-azure.txt    # Azure-compatible dependencies
├── main.py                   # Main application entry point
├── app.py                    # Alternative entry point
└── web.config               # IIS configuration (legacy)
```

## 🚨 Common Issues and Solutions

### Issue: "Neither main.py nor app.py found"
**Solution**: The enhanced `startup-azure.sh` script now:
- Searches multiple Azure paths
- Provides detailed debugging information
- Has fallback mechanisms

### Issue: "bad interpreter: No such file or directory"
**Solution**: Fixed line endings in shell scripts

### Issue: Deployment fails during build
**Solution**: Set `SCM_DO_BUILD_DURING_DEPLOYMENT=true`

## 📊 Monitoring Deployment

1. **Azure Portal**: Monitor deployment in App Service → Deployment Center
2. **Logs**: Check App Service → Log stream for real-time logs
3. **Health Check**: Visit `https://your-app.azurewebsites.net/health`

## 🎯 Next Steps

1. **Choose a deployment method** from the options above
2. **Run the deployment script** or use Azure Portal
3. **Monitor the deployment** for any issues
4. **Test your application** once deployed

## 🆘 Need Help?

If you continue to experience issues:

1. Check the **Log stream** in Azure App Service
2. Verify **Environment variables** are set correctly
3. Ensure **Startup command** points to `startup-azure.sh`
4. Check that **Python 3.11** runtime is selected

The enhanced startup script will provide much better debugging information to help identify any remaining issues. 