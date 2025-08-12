# Azure App Service Deployment Script for AI Product Trend Predictor
# Run this script from PowerShell to deploy to Azure

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,
    
    [Parameter(Mandatory=$true)]
    [string]$AppServiceName,
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "East US"
)

Write-Host "🚀 Deploying AI Product Trend Predictor to Azure..." -ForegroundColor Green
Write-Host "📋 Resource Group: $ResourceGroupName" -ForegroundColor Yellow
Write-Host "🌐 App Service: $AppServiceName" -ForegroundColor Yellow
Write-Host "📍 Location: $Location" -ForegroundColor Yellow

# Check if Azure CLI is installed
try {
    $azVersion = az version --output json | ConvertFrom-Json
    Write-Host "✅ Azure CLI version: $($azVersion.'azure-cli')" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI not found. Please install it first:" -ForegroundColor Red
    Write-Host "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli" -ForegroundColor Yellow
    exit 1
}

# Check if logged in to Azure
try {
    $account = az account show --output json | ConvertFrom-Json
    Write-Host "✅ Logged in as: $($account.user.name)" -ForegroundColor Green
} catch {
    Write-Host "🔐 Please log in to Azure first:" -ForegroundColor Yellow
    az login
}

# Create resource group if it doesn't exist
Write-Host "📁 Creating resource group..." -ForegroundColor Blue
az group create --name $ResourceGroupName --location $Location

# Create App Service plan
$planName = "$AppServiceName-plan"
Write-Host "📋 Creating App Service plan..." -ForegroundColor Blue
az appservice plan create --name $planName --resource-group $ResourceGroupName --location $Location --sku B1 --is-linux

# Create App Service
Write-Host "🌐 Creating App Service..." -ForegroundColor Blue
az webapp create --name $AppServiceName --resource-group $ResourceGroupName --plan $planName --runtime "PYTHON|3.11"

# Configure App Service
Write-Host "⚙️  Configuring App Service..." -ForegroundColor Blue
az webapp config set --name $AppServiceName --resource-group $ResourceGroupName --startup-file "bash startup-azure.sh"

# Set environment variables
Write-Host "🔧 Setting environment variables..." -ForegroundColor Blue
az webapp config appsettings set --name $AppServiceName --resource-group $ResourceGroupName --settings PYTHONPATH="/home/site/wwwroot" PYTHONUNBUFFERED="1" PORT="8000" WEBSITES_PORT="8000"

# Enable build during deployment
Write-Host "🔨 Enabling build during deployment..." -ForegroundColor Blue
az webapp config appsettings set --name $AppServiceName --resource-group $ResourceGroupName --settings SCM_DO_BUILD_DURING_DEPLOYMENT="true"

# Deploy the application
Write-Host "📦 Deploying application..." -ForegroundColor Blue
az webapp deployment source config-zip --name $AppServiceName --resource-group $ResourceGroupName --src "project-files.zip"

# Get the app URL
$appUrl = "https://$AppServiceName.azurewebsites.net"
Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
Write-Host "🌐 Your app is available at: $appUrl" -ForegroundColor Green
Write-Host "📚 API Documentation: $appUrl/docs" -ForegroundColor Green
Write-Host "💚 Health Check: $appUrl/health" -ForegroundColor Green

Write-Host "🔄 The app may take a few minutes to fully start up..." -ForegroundColor Yellow
Write-Host "📊 Monitor the deployment in Azure Portal: https://portal.azure.com" -ForegroundColor Blue 