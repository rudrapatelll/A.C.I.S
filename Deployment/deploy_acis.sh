#!/bin/bash

# ACIS Deployment Script
set -e

echo "🚀 Starting ACIS Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose availability (support both old and new versions)
COMPOSE_CMD=""
if command -v docker compose &> /dev/null; then
    COMPOSE_CMD="docker compose"
    echo "✅ Using Docker Compose v2"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo "✅ Using Docker Compose v1"
else
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads outputs models data nginx

# Build and start services
echo "🔧 Building Docker image..."
$COMPOSE_CMD build

echo "🚀 Starting ACIS application..."
$COMPOSE_CMD up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if $COMPOSE_CMD ps | grep -q "acis-app.*Up"; then
    echo "✅ ACIS is running successfully!"
    echo ""
    echo "🌐 Access your ACIS application at:"
    echo "   📱 Web Interface: http://localhost:8501"
    echo ""
    echo "📋 Available endpoints:"
    echo "   • Main Interface: http://localhost:8501"
    echo ""
    echo "📁 Default directories:"
    echo "   • Uploads: ./uploads/"
    echo "   • Outputs: ./outputs/"
    echo "   • Models: ./models/"
    echo ""
    echo "🛑 To stop the application:"
    echo "   $COMPOSE_CMD down"
else
    echo "❌ Failed to start ACIS. Check logs with:"
    echo "   $COMPOSE_CMD logs acis-app"
    exit 1
fi