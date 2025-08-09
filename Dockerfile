FROM python:3.11-slim

# Set working directory
WORKDIR /home/site/wwwroot

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements-azure.txt requirements.txt
COPY requirements-azure.txt .

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-azure.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/saved dashboard/static logs evaluation_results

# Set environment variables
ENV PYTHONPATH=/home/site/wwwroot
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Make startup script executable
RUN chmod +x startup.sh

# Expose port
EXPOSE 8000

# Health check with longer timeout for slow Azure startup
HEALTHCHECK --interval=30s --timeout=60s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application using the startup script
CMD ["./startup.sh"]