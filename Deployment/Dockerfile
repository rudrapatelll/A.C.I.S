# Use Python 3.10 slim for better compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (updated for newer Debian/Ubuntu)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    git \
    wget \
    curl \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY metrics_server.py .

# Create necessary directories
RUN mkdir -p uploads outputs models data

# Expose ports
EXPOSE 8501 8000

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Create startup script to launch both metrics server and Streamlit
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Starting ACIS services..."\n\
echo "📊 Starting Prometheus metrics server on port 8000..."\n\
python3 -c "from metrics_server import start_metrics_server_thread; import time; start_metrics_server_thread(); time.sleep(3)" &\n\
METRICS_PID=$!\n\
echo "✅ Metrics server started (PID: $METRICS_PID)"\n\
echo "🎨 Starting Streamlit app on port 8501..."\n\
streamlit run app.py --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run startup script
CMD ["/bin/bash", "/app/start.sh"]