#!/bin/bash

# ACIS System Check Script
# Verifies system requirements before deployment

echo "🔍 ACIS System Requirements Check"
echo "================================="

# Check operating system
echo ""
echo "🖥️  Operating System:"
uname -s
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ Linux detected"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✅ macOS detected"
else
    echo "⚠️  $OSTYPE may not be fully supported"
fi

# Check Docker installation
echo ""
echo "🐳 Docker Check:"
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    docker_version=$(docker --version 2>/dev/null)
    echo "   Version: $docker_version"
    
    if docker info &> /dev/null; then
        echo "✅ Docker daemon is running"
    else
        echo "❌ Docker daemon is not running"
        echo "   Please start Docker Desktop or Docker daemon"
    fi
else
    echo "❌ Docker is not installed"
    echo "   Please install Docker from https://www.docker.com/products/docker-desktop"
fi

# Check Docker Compose
echo ""
echo "📦 Docker Compose Check:"
COMPOSE_FOUND=false
if command -v docker compose &> /dev/null; then
    echo "✅ Docker Compose v2 is installed"
    compose_version=$(docker compose version 2>/dev/null)
    echo "   Version: $compose_version"
    COMPOSE_FOUND=true
elif command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose v1 is installed"
    compose_version=$(docker-compose --version 2>/dev/null)
    echo "   Version: $compose_version"
    COMPOSE_FOUND=true
else
    echo "❌ Docker Compose is not installed"
    echo "   Docker Compose should be included with Docker Desktop"
fi

# Check available memory
echo ""
echo "💾 Memory Check:"
if command -v free &> /dev/null; then
    # Linux
    memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    echo "✅ Available RAM: ${memory_gb}GB"
    if [ "$memory_gb" -lt 4 ]; then
        echo "⚠️  Warning: Less than 4GB RAM may cause performance issues"
    else
        echo "✅ Sufficient memory for ACIS"
    fi
elif command -v vm_stat &> /dev/null; then
    # macOS
    page_size=$(vm_stat | grep "page size of" | awk '{print $8}')
    pages_free=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    memory_bytes=$((pages_free * page_size))
    memory_gb=$((memory_bytes / 1024 / 1024 / 1024))
    echo "✅ Available RAM: ${memory_gb}GB"
    if [ "$memory_gb" -lt 4 ]; then
        echo "⚠️  Warning: Less than 4GB RAM may cause performance issues"
    else
        echo "✅ Sufficient memory for ACIS"
    fi
else
    echo "⚠️  Could not determine available memory"
fi

# Check disk space
echo ""
echo "💽 Disk Space Check:"
if command -v df &> /dev/null; then
    available_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    echo "✅ Available disk space: ${available_gb}GB"
    if [ "$available_gb" -lt 10 ]; then
        echo "⚠️  Warning: Less than 10GB disk space may cause issues"
    else
        echo "✅ Sufficient disk space"
    fi
else
    echo "⚠️  Could not determine available disk space"
fi

# Check internet connectivity
echo ""
echo "🌐 Internet Connectivity:"
if ping -c 1 google.com &> /dev/null; then
    echo "✅ Internet connection available"
else
    echo "⚠️  No internet connection detected"
    echo "   Initial setup may require internet access"
fi

# Check port availability
echo ""
echo "🔌 Port Availability:"
if netstat -ln 2>/dev/null | grep -q ":8501"; then
    echo "⚠️  Port 8501 is already in use"
    echo "   ACIS needs port 8501 to be available"
else
    echo "✅ Port 8501 is available"
fi

# GPU Check (optional)
echo ""
echo "🎮 GPU Check (Optional):"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
else
    echo "ℹ️  No NVIDIA GPU detected (optional)"
    echo "   CPU mode will be used for detection"
fi

# Final summary
echo ""
echo "📋 Summary:"
echo "==========="

checks_passed=0
total_checks=4

# Count passed checks
command -v docker &> /dev/null && checks_passed=$((checks_passed + 1))
command -v docker-compose &> /dev/null && checks_passed=$((checks_passed + 1))
ping -c 1 google.com &> /dev/null && checks_passed=$((checks_passed + 1))
! netstat -ln 2>/dev/null | grep -q ":8501" && checks_passed=$((checks_passed + 1))

if [ "$checks_passed" -eq "$total_checks" ]; then
    echo "✅ All system checks passed!"
    echo "🚀 Ready to deploy ACIS"
    echo ""
    echo "Next steps:"
    echo "1. Run: ./deploy_acis.sh"
    echo "2. Open: http://localhost:8501"
elif [ "$checks_passed" -ge 2 ]; then
    echo "⚠️  Some checks failed, but deployment may still work"
    echo "📖 Review warnings above and run: ./deploy_acis.sh"
else
    echo "❌ Multiple system requirements failed"
    echo "🔧 Please fix the issues above before deployment"
    echo "📖 See README.md for detailed installation instructions"
fi

echo ""
echo "ℹ️  For detailed setup instructions, see README.md"