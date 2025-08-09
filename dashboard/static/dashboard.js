// Global variables
let currentData = {};
let charts = {};
let currentTab = 'dashboard';

// API base URL
const API_BASE = window.location.origin;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

function initializeDashboard() {
    console.log('Initializing AI Trend Prediction Dashboard...');
    
    // Load initial data
    loadDashboardData();
    
    // Set up event listeners
    setupEventListeners();
    
    // Show dashboard tab by default
    showTab('dashboard');
}

function setupEventListeners() {
    // Prediction form
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    
    // Upload form
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleUploadSubmit);
    }
    
    // Load Superstore data form
    const generateForm = document.getElementById('generate-form');
    if (generateForm) {
        generateForm.addEventListener('submit', handleGenerateSubmit);
    }
    
    // Search and filter functionality
    const productSearch = document.getElementById('product-search');
    if (productSearch) {
        productSearch.addEventListener('input', filterProducts);
    }
    
    const categoryFilter = document.getElementById('category-filter');
    if (categoryFilter) {
        categoryFilter.addEventListener('change', filterProducts);
    }
    
    const sortProducts = document.getElementById('sort-products');
    if (sortProducts) {
        sortProducts.addEventListener('change', filterProducts);
    }
    
    // Radio button toggle for sample options
    document.querySelectorAll('input[name="reload-option"]').forEach(radio => {
        radio.addEventListener('change', function() {
            const sampleOptions = document.getElementById('sample-options');
            if (this.value === 'sample') {
                sampleOptions.style.display = 'block';
            } else {
                sampleOptions.style.display = 'none';
            }
        });
    });
}

// Tab navigation
function showTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-pane');
    tabs.forEach(tab => tab.style.display = 'none');
    
    // Remove active class from all nav links
    const navLinks = document.querySelectorAll('.sidebar .nav-link');
    navLinks.forEach(link => link.classList.remove('active'));
    
    // Show selected tab
    const selectedTab = document.getElementById(`${tabName}-tab`);
    if (selectedTab) {
        selectedTab.style.display = 'block';
        currentTab = tabName;
        
        // Add active class to corresponding nav link
        const activeLink = document.querySelector(`[onclick="showTab('${tabName}')"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
        
        // Load tab-specific data
        loadTabData(tabName);
    }
}

async function loadTabData(tabName) {
    switch (tabName) {
        case 'dashboard':
            await loadDashboardData();
            break;
        case 'predictions':
            await loadProductsForSelect();
            break;
        case 'products':
            await loadProducts();
            break;
        case 'analytics':
            await loadAnalytics();
            break;
        case 'alerts':
            await loadAlerts();
            break;
        case 'data':
            await loadDatasetInfo();
            break;
        case 'model':
            await loadModelStatus();
            break;
    }
}

// API calls
async function apiCall(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`API call failed for ${endpoint}:`, error);
        showError(`Failed to ${endpoint}: ${error.message}`);
        throw error;
    }
}

// Dashboard data loading - optimized for faster UX
async function loadDashboardData() {
    try {
        showLoading('Loading dashboard data...');
        
        // First, load basic metrics super fast
        try {
            const quickMetrics = await apiCall('/metrics/quick');
            if (quickMetrics.dataset_available) {
                // Show basic metrics immediately
                updateMetrics(quickMetrics);
                hideLoading(); // Hide main loading - no secondary popup!
                console.log('Quick metrics loaded, preparing data tables...');
            }
        } catch (e) {
            console.log('Quick metrics failed, continuing with full load');
        }
        
        // Try to load overview data, but don't block if it fails
        let overview = null;
        try {
            overview = await apiCall('/analytics/overview');
            currentData.overview = overview;
            
            // Update metrics immediately
            updateMetrics(overview.summary);
        } catch (e) {
            console.error('Analytics overview failed:', e);
            // Keep the quick metrics that were already loaded
            showError('Advanced analytics unavailable. Basic metrics are still shown.', 4000);
        }
        
        hideLoading(); // Hide loading after metrics are shown
        
        // Create charts for dashboard
        setTimeout(() => {
            try {
                if (overview) {
                    console.log('Creating dashboard charts...');
                    
                    // Create sales trends chart
                    createSalesChart(overview.daily_trends || []);
                    
                    // Create category distribution chart
                    createCategoryChart(overview.category_performance || {});
                    
                    console.log('Dashboard charts created successfully');
                }
                
                // Force hide any remaining loading popups
                hideLoading();
                
                // Clear any loading timeouts that might be running
                if (loadingTimeoutId) {
                    clearTimeout(loadingTimeoutId);
                    loadingTimeoutId = null;
                }
                
                showSuccess('Dashboard loaded successfully!', 3000);
            } catch (e) {
                // Force hide any remaining loading popups
                hideLoading();
                
                // Clear any loading timeouts that might be running
                if (loadingTimeoutId) {
                    clearTimeout(loadingTimeoutId);
                    loadingTimeoutId = null;
                }
                
                console.error('Chart creation failed:', e);
                showError('Charts failed to load, but dashboard is ready to use.', 3000);
            }
        }, 100);
        
        // Load alerts in background
        setTimeout(() => {
            loadRecentAlerts().catch(e => console.error('Alerts failed:', e));
        }, 200);
        
    } catch (error) {
        hideLoading();
        showError('Failed to load dashboard data: ' + error.message);
        // Show basic fallback data
        updateMetrics({
            total_products: '-',
            total_sales: '-', 
            total_revenue: '-',
            avg_order_value: '-'
        });
    }
}

function updateMetrics(summary) {
    document.getElementById('total-products').textContent = summary.total_products || '-';
    document.getElementById('total-sales').textContent = formatNumber(summary.total_sales) || '-';
    document.getElementById('total-revenue').textContent = formatCurrency(summary.total_revenue) || '-';
    document.getElementById('avg-order-value').textContent = formatCurrency(summary.avg_order_value) || '-';
}

async function loadRecentAlerts() {
    try {
        const alertsData = await apiCall('/alerts');
        const alertsContainer = document.getElementById('recent-alerts');
        
        if (alertsData.alerts && alertsData.alerts.length > 0) {
            const recentAlerts = alertsData.alerts.slice(0, 3);
            alertsContainer.innerHTML = recentAlerts.map(alert => `
                <div class="alert-item alert-${alert.severity}">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="mb-1">${alert.product_id}</h6>
                            <p class="mb-1">${alert.message}</p>
                            <small class="text-muted">Confidence: ${(alert.confidence * 100).toFixed(1)}%</small>
                        </div>
                        <span class="badge bg-${getSeverityColor(alert.severity)}">${alert.severity}</span>
                    </div>
                </div>
            `).join('');
        } else {
            alertsContainer.innerHTML = '<p class="text-muted">No recent alerts</p>';
        }
    } catch (error) {
        document.getElementById('recent-alerts').innerHTML = '<p class="text-danger">Failed to load alerts</p>';
    }
}

// Products data loading
async function loadProducts() {
    try {
        showElementLoading('products-list');
        
        const productsData = await apiCall('/products');
        currentData.products = productsData.products;
        
        // Populate category filter
        populateCategoryFilter(productsData.products);
        
        // Display products
        displayProducts(productsData.products);
        
        hideElementLoading('products-list');
    } catch (error) {
        hideElementLoading('products-list');
        showError('Failed to load products');
    }
}

async function loadProductsForSelect() {
    try {
        const productsData = await apiCall('/products');
        const productSelect = document.getElementById('product-select');
        
        productSelect.innerHTML = '<option value="">Select a product...</option>';
        
        productsData.products.forEach(product => {
            const option = document.createElement('option');
            option.value = product.product_id;
            option.textContent = `${product.product_name || 'N/A'} (${product.product_id}) - ${product.category}`;
            productSelect.appendChild(option);
        });
    } catch (error) {
        showError('Failed to load products for selection');
    }
}

function populateCategoryFilter(products) {
    const categories = [...new Set(products.map(p => p.category))];
    const categoryFilter = document.getElementById('category-filter');
    
    categoryFilter.innerHTML = '<option value="">All Categories</option>';
    categories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        categoryFilter.appendChild(option);
    });
}

function displayProducts(products) {
    const container = document.getElementById('products-list');
    
    if (products.length === 0) {
        container.innerHTML = '<p class="text-muted">No products found</p>';
        return;
    }
    
    container.innerHTML = products.map(product => `
        <div class="product-card" onclick="viewProductDetails('${product.product_id}')">
            <div class="row align-items-center">
                <div class="col-md-4">
                    <h6 class="mb-1">${product.product_name || 'N/A'}</h6>
                    <p class="mb-1 text-muted">${product.product_id}</p>
                    <div class="mb-2">
                        <span class="badge bg-primary">${product.category}</span>
                        <span class="badge bg-info">${product.sub_category}</span>
                        <span class="badge bg-secondary">${product.brand}</span>
                    </div>
                    <small class="text-muted">
                        <i class="fas fa-map-marker-alt me-1"></i>${product.region} - ${product.state}
                    </small>
                </div>
                <div class="col-md-2">
                    <div class="text-center">
                        <div class="h6 mb-0">${formatNumber(product.total_sales)}</div>
                        <small class="text-muted">Sales</small>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="text-center">
                        <div class="h6 mb-0">${formatCurrency(product.total_revenue)}</div>
                        <small class="text-muted">Revenue</small>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="text-center">
                        <div class="h6 mb-0 ${product.total_profit >= 0 ? 'text-success' : 'text-danger'}">${formatCurrency(product.total_profit)}</div>
                        <small class="text-muted">Profit</small>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="text-center">
                        <div class="h6 mb-0 ${product.profit_margin >= 0 ? 'text-success' : 'text-danger'}">${product.profit_margin}%</div>
                        <small class="text-muted">Margin</small>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

function filterProducts() {
    if (!currentData.products) return;
    
    const searchTerm = document.getElementById('product-search').value.toLowerCase();
    const categoryFilter = document.getElementById('category-filter').value;
    const sortBy = document.getElementById('sort-products').value;
    
    let filtered = currentData.products.filter(product => {
        const matchesSearch = (product.product_name || '').toLowerCase().includes(searchTerm) ||
                            product.product_id.toLowerCase().includes(searchTerm) ||
                            (product.sub_category || '').toLowerCase().includes(searchTerm);
        const matchesCategory = !categoryFilter || product.category === categoryFilter;
        
        return matchesSearch && matchesCategory;
    });
    
    // Sort products
    filtered.sort((a, b) => {
        switch (sortBy) {
            case 'product_name':
                return (a.product_name || '').localeCompare(b.product_name || '');
            case 'category':
                return (a.category || '').localeCompare(b.category || '');
            case 'total_sales':
                return (b.total_sales || 0) - (a.total_sales || 0);
            case 'total_revenue':
                return (b.total_revenue || 0) - (a.total_revenue || 0);
            case 'total_profit':
                return (b.total_profit || 0) - (a.total_profit || 0);
            case 'profit_margin':
                return (b.profit_margin || 0) - (a.profit_margin || 0);
            default:
                return 0;
        }
    });
    
    displayProducts(filtered);
}

// Predictions
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const productId = document.getElementById('product-select').value;
    const daysAhead = document.getElementById('days-ahead').value;
    
    if (!productId) {
        showError('Please select a product');
        return;
    }
    
    try {
        showLoading('Generating predictions...');
        
        const prediction = await apiCall('/predict', {
            method: 'POST',
            body: JSON.stringify({
                product_id: productId,
                days_ahead: parseInt(daysAhead),
                include_confidence: true
            })
        });
        
        displayPredictionResults(prediction);
        hideLoading();
        
    } catch (error) {
        hideLoading();
        showError('Failed to generate prediction');
    }
}

function displayPredictionResults(prediction) {
    document.getElementById('prediction-placeholder').style.display = 'none';
    document.getElementById('prediction-results').style.display = 'block';
    
    const content = document.getElementById('prediction-content');
    content.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <div class="text-center">
                    <h4 class="trend-${prediction.trend_direction}">${prediction.trend_direction.toUpperCase()}</h4>
                    <small>Trend Direction</small>
                </div>
            </div>
            <div class="col-md-4">
                <div class="text-center">
                    <h4>${(prediction.trend_strength * 100).toFixed(1)}%</h4>
                    <small>Trend Strength</small>
                </div>
            </div>
            <div class="col-md-4">
                <div class="text-center">
                    <h4>${(prediction.model_accuracy * 100).toFixed(1)}%</h4>
                    <small>Model Accuracy</small>
                </div>
            </div>
        </div>
    `;
    
    // Create prediction chart
    createPredictionChart(prediction);
}

// Analytics
async function loadAnalytics() {
    try {
        showElementLoading('analytics-tbody');
        
        const overview = await apiCall('/analytics/overview');
        
        // Create brand chart
        createBrandChart(overview.brand_performance);
        
        // Create monthly chart (simplified for demo)
        createMonthlyChart(overview.daily_trends);
        
        // Update analytics table
        updateAnalyticsTable(overview.category_performance);
        
        hideElementLoading('analytics-tbody');
    } catch (error) {
        hideElementLoading('analytics-tbody');
        showError('Failed to load analytics');
    }
}

function updateAnalyticsTable(categoryData) {
    const tbody = document.getElementById('analytics-tbody');
    
    tbody.innerHTML = Object.entries(categoryData).map(([category, data]) => `
        <tr>
            <td>${category}</td>
            <td>${formatNumber(data.quantity_sold)}</td>
            <td>${formatCurrency(data.revenue)}</td>
            <td>${formatCurrency(data.revenue / data.quantity_sold)}</td>
            <td><span class="badge bg-success">+5.2%</span></td>
        </tr>
    `).join('');
}

// Alerts
async function loadAlerts() {
    try {
        showElementLoading('alerts-list');
        
        const alertsData = await apiCall('/alerts');
        currentData.alerts = alertsData.alerts;
        
        displayAlerts(alertsData.alerts);
        hideElementLoading('alerts-list');
    } catch (error) {
        hideElementLoading('alerts-list');
        showError('Failed to load alerts');
    }
}

function displayAlerts(alerts) {
    const container = document.getElementById('alerts-list');
    
    if (alerts.length === 0) {
        container.innerHTML = '<p class="text-muted">No alerts found</p>';
        return;
    }
    
    container.innerHTML = alerts.map(alert => `
        <div class="alert-item alert-${alert.severity}">
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <h6 class="mb-1">${alert.product_id}</h6>
                    <p class="mb-1">${alert.message}</p>
                    <small class="text-muted">
                        <i class="fas fa-clock me-1"></i>
                        ${formatDate(alert.detected_at)} | 
                        Confidence: ${(alert.confidence * 100).toFixed(1)}%
                    </small>
                </div>
                <div>
                    <span class="badge bg-${getSeverityColor(alert.severity)} mb-2">${alert.severity}</span>
                    <br>
                    <span class="badge bg-info">${alert.alert_type}</span>
                </div>
            </div>
        </div>
    `).join('');
}

function filterAlerts(severity) {
    if (!currentData.alerts) return;
    
    // Update button states
    const buttons = document.querySelectorAll('.btn-group .btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    let filtered = currentData.alerts;
    if (severity !== 'all') {
        filtered = currentData.alerts.filter(alert => alert.severity === severity);
    }
    
    displayAlerts(filtered);
}

async function refreshAlerts() {
    await loadAlerts();
}

// Data Management
async function loadDatasetInfo() {
    try {
        const overview = await apiCall('/analytics/overview');
        const health = await apiCall('/health');
        
        const info = document.getElementById('dataset-info');
        info.innerHTML = `
            <div class="row mb-3">
                <div class="col-md-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title"><i class="fas fa-boxes me-2"></i>Dataset Overview</h6>
                            <div class="row">
                                <div class="col-6">
                                    <strong>Total Products:</strong><br>
                                    <span class="h5 text-primary">${overview.summary.total_products.toLocaleString()}</span>
                                </div>
                                <div class="col-6">
                                    <strong>Total Sales:</strong><br>
                                    <span class="h5 text-success">${formatNumber(overview.summary.total_sales)}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title"><i class="fas fa-chart-line me-2"></i>Financial Summary</h6>
                            <div class="row">
                                <div class="col-6">
                                    <strong>Total Revenue:</strong><br>
                                    <span class="h5 text-success">${formatCurrency(overview.summary.total_revenue)}</span>
                                </div>
                                <div class="col-6">
                                    <strong>Avg Order Value:</strong><br>
                                    <span class="h5 text-info">${formatCurrency(overview.summary.avg_order_value)}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mb-3">
                <div class="col-md-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title"><i class="fas fa-calendar me-2"></i>Date Range</h6>
                            <strong>From:</strong> ${formatDate(overview.summary.date_range.start)}<br>
                            <strong>To:</strong> ${formatDate(overview.summary.date_range.end)}
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title"><i class="fas fa-brain me-2"></i>Model Status</h6>
                            <span class="badge bg-${health.model_trained ? 'success' : 'warning'} fs-6">
                                <i class="fas fa-${health.model_trained ? 'check' : 'exclamation-triangle'} me-1"></i>
                                ${health.model_trained ? 'Model Trained' : 'Model Not Trained'}
                            </span>
                            ${health.model_trained ? '<br><small class="text-muted">Ready for predictions</small>' : '<br><small class="text-muted">Train model to enable predictions</small>'}
                        </div>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        document.getElementById('dataset-info').innerHTML = '<p class="text-danger">Failed to load dataset info</p>';
    }
}

async function handleUploadSubmit(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showLoading('Uploading data...');
        
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const result = await response.json();
        showSuccess('Data uploaded successfully');
        
        // Refresh dataset info
        await loadDatasetInfo();
        fileInput.value = '';
        
        hideLoading();
    } catch (error) {
        hideLoading();
        showError('Failed to upload data');
    }
}

async function handleGenerateSubmit(event) {
    event.preventDefault();
    
    const reloadOption = document.querySelector('input[name="reload-option"]:checked').value;
    
    try {
        if (reloadOption === 'complete') {
            showLoading('Loading complete Superstore dataset...', 'Loading all 1,610+ products and 3,939+ records...');
            await apiCall('/reload-superstore-data');
        } else {
            const numProducts = document.getElementById('num-products').value;
            showLoading(`Loading sample Superstore dataset...`, `Loading top ${numProducts} products...`);
            await apiCall(`/reload-superstore-data?n_products=${numProducts}&days=365`);
        }
        
        showSuccess('Superstore data loaded successfully');
        await loadDatasetInfo();
        
        hideLoading();
    } catch (error) {
        hideLoading();
        showError('Failed to load Superstore data: ' + error.message);
    }
}

async function downloadData() {
    try {
        const response = await fetch(`${API_BASE}/download-sample-data`);
        
        if (!response.ok) {
            throw new Error('Download failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ecommerce_data.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
    } catch (error) {
        showError('Failed to download data');
    }
}

// Model Management
async function loadModelStatus() {
    try {
        showElementLoading('model-metrics');
        
        const status = await apiCall('/model/status');
        
        // Update status badge
        const statusElement = document.getElementById('model-status');
        statusElement.innerHTML = `
            <span class="badge bg-${status.trained ? 'success' : 'warning'}">
                ${status.trained ? 'Model Trained' : 'Not Trained'}
            </span>
        `;
        
        if (status.trained && status.model_metrics) {
            displayModelMetrics(status.model_metrics);
            displayFeatureImportance(status.feature_importance);
        } else {
            document.getElementById('model-metrics').innerHTML = '<p class="text-muted">No model metrics available</p>';
        }
        
        hideElementLoading('model-metrics');
    } catch (error) {
        hideElementLoading('model-metrics');
        showError('Failed to load model status');
    }
}

function displayModelMetrics(metrics) {
    const container = document.getElementById('model-metrics');
    
    container.innerHTML = Object.entries(metrics).map(([modelName, metric]) => `
        <div class="mb-4">
            <h6>${modelName.toUpperCase()}</h6>
            <div class="row">
                <div class="col-md-3">
                    <div class="text-center">
                        <div class="h6 mb-0">${metric.mae.toFixed(3)}</div>
                        <small class="text-muted">MAE</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="text-center">
                        <div class="h6 mb-0">${metric.rmse.toFixed(3)}</div>
                        <small class="text-muted">RMSE</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="text-center">
                        <div class="h6 mb-0">${(metric.r2_score * 100).toFixed(1)}%</div>
                        <small class="text-muted">R² Score</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="text-center">
                        <div class="h6 mb-0">${metric.mape.toFixed(1)}%</div>
                        <small class="text-muted">MAPE</small>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

async function trainModel() {
    try {
        showLoading('Training model...', 'This may take several minutes. The model will train in the background.');
        
        await apiCall('/train', { method: 'POST' });
        
        showSuccess('Model training started in background');
        
        // Wait a bit and refresh status
        setTimeout(async () => {
            await loadModelStatus();
        }, 2000);
        
        hideLoading();
    } catch (error) {
        hideLoading();
        showError('Failed to start model training');
    }
}

// Chart creation functions
function createSalesChart(dailyTrends) {
    const ctx = document.getElementById('sales-chart');
    if (!ctx) return;
    
    if (charts.sales) {
        charts.sales.destroy();
    }
    
    const labels = dailyTrends.map(d => formatDate(d.date));
    const salesData = dailyTrends.map(d => d.quantity_sold);
    const revenueData = dailyTrends.map(d => d.revenue);
    
    charts.sales = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sales Quantity',
                data: salesData,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.1,
                yAxisID: 'y'
            }, {
                label: 'Revenue',
                data: revenueData,
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                tension: 0.1,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Quantity Sold'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Revenue ($)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}

function createCategoryChart(categoryData) {
    const ctx = document.getElementById('category-chart');
    if (!ctx) return;
    
    if (charts.category) {
        charts.category.destroy();
    }
    
    const labels = Object.keys(categoryData);
    const data = Object.values(categoryData).map(d => d.revenue);
    
    charts.category = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#3498db', '#27ae60', '#f39c12', '#e74c3c',
                    '#9b59b6', '#34495e', '#1abc9c', '#e67e22'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                }
            }
        }
    });
}

function createPredictionChart(prediction) {
    const ctx = document.getElementById('prediction-chart');
    if (!ctx) return;
    
    if (charts.prediction) {
        charts.prediction.destroy();
    }
    
    const labels = prediction.predictions.map(p => formatDate(p.date));
    const predictions = prediction.predictions.map(p => p.predicted_quantity);
    const lowerBounds = prediction.confidence_intervals?.map(c => c.lower_bound) || [];
    const upperBounds = prediction.confidence_intervals?.map(c => c.upper_bound) || [];
    
    charts.prediction = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Predicted Sales',
                data: predictions,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.1
            }, {
                label: 'Lower Bound',
                data: lowerBounds,
                borderColor: '#e74c3c',
                backgroundColor: 'transparent',
                borderDash: [5, 5],
                tension: 0.1
            }, {
                label: 'Upper Bound',
                data: upperBounds,
                borderColor: '#27ae60',
                backgroundColor: 'transparent',
                borderDash: [5, 5],
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Predicted Quantity'
                    }
                }
            }
        }
    });
}

function createCategoryPieChart(categoryData) {
    const ctx = document.getElementById('category-pie-chart');
    if (!ctx || !categoryData) return;
    
    // Destroy existing chart
    if (charts.categoryPie) {
        charts.categoryPie.destroy();
    }
    
    const categories = Object.keys(categoryData);
    const revenues = categories.map(category => categoryData[category].revenue);
    const quantities = categories.map(category => categoryData[category].quantity_sold);
    
    // Color palette for categories
    const colors = [
        'rgba(255, 99, 132, 0.8)',   // Red
        'rgba(54, 162, 235, 0.8)',   // Blue
        'rgba(255, 205, 86, 0.8)',   // Yellow
        'rgba(75, 192, 192, 0.8)',   // Green
        'rgba(153, 102, 255, 0.8)',  // Purple
        'rgba(255, 159, 64, 0.8)'    // Orange
    ];
    
    charts.categoryPie = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: categories,
            datasets: [{
                data: revenues,
                backgroundColor: colors.slice(0, categories.length),
                borderColor: colors.slice(0, categories.length).map(color => color.replace('0.8', '1')),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const category = context.label;
                            const revenue = context.parsed;
                            const quantity = quantities[context.dataIndex];
                            const percentage = ((revenue / revenues.reduce((a, b) => a + b, 0)) * 100).toFixed(1);
                            return `${category}: $${revenue.toLocaleString()} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

function createBrandChart(brandData) {
    const ctx = document.getElementById('brand-chart');
    if (!ctx) return;
    
    if (charts.brand) {
        charts.brand.destroy();
    }
    
    const labels = Object.keys(brandData).slice(0, 10); // Top 10 brands
    const data = labels.map(brand => brandData[brand].revenue);
    
    charts.brand = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Revenue',
                data: data,
                backgroundColor: '#3498db'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Revenue ($)'
                    }
                }
            }
        }
    });
}

function createMonthlyChart(dailyTrends) {
    const ctx = document.getElementById('monthly-chart');
    if (!ctx) return;
    
    if (charts.monthly) {
        charts.monthly.destroy();
    }
    
    // Aggregate by month for demo
    const monthlyData = {};
    dailyTrends.forEach(d => {
        const month = d.date.substring(0, 7); // YYYY-MM
        if (!monthlyData[month]) {
            monthlyData[month] = 0;
        }
        monthlyData[month] += d.revenue;
    });
    
    const labels = Object.keys(monthlyData).sort();
    const data = labels.map(month => monthlyData[month]);
    
    charts.monthly = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Monthly Revenue',
                data: data,
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Revenue ($)'
                    }
                }
            }
        }
    });
}

function displayFeatureImportance(featureImportance) {
    if (!featureImportance || Object.keys(featureImportance).length === 0) {
        document.getElementById('feature-importance').innerHTML = '<p class="text-muted">No feature importance data available</p>';
        return;
    }
    
    // Use XGBoost importance for display
    const xgbImportance = featureImportance.xgboost || {};
    const features = Object.keys(xgbImportance).slice(0, 10); // Top 10 features
    const importance = features.map(f => xgbImportance[f]);
    
    const ctx = document.getElementById('feature-chart');
    if (!ctx) return;
    
    if (charts.feature) {
        charts.feature.destroy();
    }
    
    charts.feature = new Chart(ctx, {
        type: 'horizontalBar',
        data: {
            labels: features,
            datasets: [{
                label: 'Importance',
                data: importance,
                backgroundColor: '#3498db'
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Feature Importance'
                    }
                }
            }
        }
    });
}

// Utility functions
function formatNumber(num) {
    if (num === null || num === undefined) return '-';
    return new Intl.NumberFormat().format(num);
}

function formatCurrency(amount) {
    if (amount === null || amount === undefined) return '-';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatDate(dateString) {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleDateString();
}

function getSeverityColor(severity) {
    switch (severity) {
        case 'high': return 'danger';
        case 'medium': return 'warning';
        case 'low': return 'success';
        default: return 'secondary';
    }
}

// Loading and error handling
let loadingTimeoutId = null;

function showLoading(text = 'Loading...', details = 'Please wait while we process your request.') {
    // Clear any existing timeout
    if (loadingTimeoutId) {
        clearTimeout(loadingTimeoutId);
        loadingTimeoutId = null;
    }
    
    // If details is a number, treat it as timeout milliseconds
    let timeoutMs = null;
    let detailsText = details;
    
    if (typeof details === 'number') {
        timeoutMs = details;
        detailsText = 'Please wait while we process your request.';
    }
    
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-details').textContent = detailsText;
    
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
    
    // Auto-hide after timeout if specified
    if (timeoutMs) {
        loadingTimeoutId = setTimeout(() => {
            hideLoading();
            loadingTimeoutId = null;
        }, timeoutMs);
    }
}

function hideLoading() {
    // Clear any pending timeout
    if (loadingTimeoutId) {
        clearTimeout(loadingTimeoutId);
        loadingTimeoutId = null;
    }
    
    // Force hide loading modal - multiple attempts to handle stuck modals
    const loadingModal = document.getElementById('loadingModal');
    if (loadingModal) {
        const modal = bootstrap.Modal.getInstance(loadingModal);
        if (modal) {
            modal.hide();
        }
        
        // Force remove backdrop and modal classes if still showing
        setTimeout(() => {
            loadingModal.classList.remove('show');
            loadingModal.style.display = 'none';
            
            // Remove any leftover backdrops
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => backdrop.remove());
            
            // Remove modal-open class from body
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
        }, 100);
    }
}

function showElementLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="loading-spinner" style="display: block;">
                <i class="fas fa-spinner fa-spin fa-2x"></i>
                <p>Loading...</p>
            </div>
        `;
    }
}

function hideElementLoading(elementId) {
    const spinner = document.querySelector(`#${elementId} .loading-spinner`);
    if (spinner) {
        spinner.style.display = 'none';
    }
}

function showError(message, autoHideMs = null) {
    // Create and show error toast
    const toast = document.createElement('div');
    toast.className = 'toast align-items-center text-white bg-danger border-0 position-fixed top-0 end-0 m-3';
    toast.style.zIndex = '9999';
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-exclamation-circle me-2"></i>${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    
    // Configure toast options
    const toastOptions = autoHideMs ? { delay: autoHideMs } : {};
    const bsToast = new bootstrap.Toast(toast, toastOptions);
    bsToast.show();
    
    // Auto-hide if specified
    if (autoHideMs) {
        setTimeout(() => {
            bsToast.hide();
        }, autoHideMs);
    }
    
    // Remove from DOM after hiding
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

function showSuccess(message, autoHideMs = null) {
    // Create and show success toast
    const toast = document.createElement('div');
    toast.className = 'toast align-items-center text-white bg-success border-0 position-fixed top-0 end-0 m-3';
    toast.style.zIndex = '9999';
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-check-circle me-2"></i>${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    
    // Configure toast options
    const toastOptions = autoHideMs ? { delay: autoHideMs } : {};
    const bsToast = new bootstrap.Toast(toast, toastOptions);
    bsToast.show();
    
    // Auto-hide if specified
    if (autoHideMs) {
        setTimeout(() => {
            bsToast.hide();
        }, autoHideMs);
    }
    
    // Remove from DOM after hiding
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

// Simple chart replacements that load instantly
function createSimpleSalesTable(dailyTrends) {
    const salesChartContainer = document.getElementById('sales-chart');
    if (!salesChartContainer || !dailyTrends || dailyTrends.length === 0) {
        return;
    }
    
    // Take last 7 days only
    const recentTrends = dailyTrends.slice(-7);
    
    salesChartContainer.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>Recent Sales Trends (Last 7 Days)</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped table-sm">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Sales Quantity</th>
                                <th>Revenue</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${recentTrends.map(item => `
                                <tr>
                                    <td>${new Date(item.date).toLocaleDateString()}</td>
                                    <td>${formatNumber(item.quantity_sold)}</td>
                                    <td>${formatCurrency(item.revenue)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <small class="text-muted">
                    <i class="fas fa-info-circle me-1"></i>
                    Charts temporarily disabled for better performance. Data is displayed in table format.
                </small>
            </div>
        </div>
    `;
}

function createSimpleCategoryList(categoryPerformance) {
    const categoryChartContainer = document.getElementById('category-chart');
    if (!categoryChartContainer || !categoryPerformance) {
        return;
    }
    
    const categories = Object.entries(categoryPerformance).slice(0, 5);
    
    categoryChartContainer.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="fas fa-chart-pie me-2"></i>Top Category Performance</h5>
            </div>
            <div class="card-body">
                <div class="list-group list-group-flush">
                    ${categories.map(([category, data], index) => `
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <div>
                                <h6 class="mb-1">${category}</h6>
                                <small class="text-muted">Sales: ${formatNumber(data.quantity_sold)}</small>
                            </div>
                            <div class="text-end">
                                <span class="badge bg-primary rounded-pill">${formatCurrency(data.revenue)}</span>
                                <br>
                                <small class="text-muted">#${index + 1}</small>
                            </div>
                        </div>
                    `).join('')}
                </div>
                <small class="text-muted mt-2 d-block">
                    <i class="fas fa-info-circle me-1"></i>
                    Charts temporarily disabled for better performance. Data is displayed in list format.
                </small>
            </div>
        </div>
    `;
}

// Global functions for onclick handlers
function refreshData() {
    loadTabData(currentTab);
}

async function reloadSuperstoreData() {
    try {
        showLoading('Reloading complete Superstore dataset...', 'Loading all 9800+ records from the Superstore dataset.');
        
        // Reload complete Superstore dataset
        const result = await apiCall('/reload-superstore-data');
        
        showSuccess(`Complete Superstore dataset reloaded! ${result.dataset_info.total_products} products with ${result.shape[0]} records.`);
        
        // Refresh current tab
        loadTabData(currentTab);
        
        hideLoading();
    } catch (error) {
        hideLoading();
        showError('Failed to reload Superstore data: ' + error.message);
    }
}

function viewProductDetails(productId) {
    // Switch to predictions tab and auto-select the product
    showTab('predictions');
    
    setTimeout(() => {
        const productSelect = document.getElementById('product-select');
        if (productSelect) {
            productSelect.value = productId;
        }
    }, 100);
}