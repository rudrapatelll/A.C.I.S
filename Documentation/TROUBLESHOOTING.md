# ACIS Docker Deployment - Troubleshooting Guide

## 🔧 Fixed Issues

Based on your deployment errors, I've fixed several compatibility issues:

### 1. Docker Compose Command Issue
**Problem**: `zsh: command not found: docker-compose`

**Solution**: The newer Docker installation uses `docker compose` (v2) instead of `docker-compose` (v1).

**Status**: ✅ Fixed in updated deployment script (`deploy_acis.sh`)

### 2. Package Availability Error
**Problem**: 
```
E: Package 'libgl1-mesa-glx' is not available, but is referred to by another package.
E: Package 'libgl1-mesa-glx' has no installation candidate
```

**Solution**: 
- Changed base image from `python:3.10-slim` to `python:3.10` (full image with better ARM64 support)
- Added CPU-specific PyTorch installation for better Apple Silicon compatibility
- Added `curl` for health checks

**Status**: ✅ Fixed in updated `Dockerfile` and `requirements.txt`

### 3. Updated Deployment Scripts
- Both `deploy_acis.sh` and `check_system.sh` now support both `docker compose` and `docker-compose`
- Better ARM64 compatibility for Mac Apple Silicon machines

## 🚀 Quick Deployment Steps

1. **Download/Extract the updated files** from your workspace

2. **Make scripts executable**:
   ```bash
   chmod +x deploy_acis.sh check_system.sh
   ```

3. **Run system check** (optional but recommended):
   ```bash
   ./check_system.sh
   ```

4. **Deploy ACIS**:
   ```bash
   ./deploy_acis.sh
   ```

5. **Access the application**:
   Open browser to: http://localhost:8501

## 🔍 If You Still Get Errors

### Error: "Docker daemon is not running"
**Solution**: 
- Start Docker Desktop application
- Wait for Docker to fully start before running deploy script

### Error: "Port 8501 is already in use"
**Solution**:
```bash
# Stop any existing containers using port 8501
docker-compose down
# Or if using docker compose v2:
docker compose down
```

### Error: "Permission denied"
**Solution**:
```bash
# Add your user to docker group
sudo usermod -aG docker $USER
# Log out and log back in for changes to take effect
```

### Error: "No space left on device"
**Solution**:
```bash
# Clean up Docker
docker system prune -a
# Or clear specific containers/images
docker system prune --volumes
```

## 📱 Access Points After Deployment

- **Main Interface**: http://localhost:8501
- **Health Check**: http://localhost:8501/_stcore/health
- **Docker Status**: `docker-compose ps` or `docker compose ps`

## 🛑 Stopping the Application

```bash
# Using docker-compose v1:
docker-compose down

# Using docker compose v2:
docker compose down
```

## 🔍 Viewing Logs

```bash
# View all logs:
docker-compose logs -f

# View specific service logs:
docker-compose logs -f acis-app

# Or if using docker compose v2:
docker compose logs -f acis-app
```

## 📋 System Requirements

- **macOS** (any recent version)
- **Docker Desktop** (latest version)
- **4GB+ RAM** available
- **2GB+ free disk space**
- **Internet connection** (for initial setup)

## 🎯 Next Steps After Deployment

1. **Upload your YOLOv8 model** (.pt file) via the web interface
2. **Configure confidence threshold** (default: 0.18)
3. **Upload video files** (MP4, AVI, MOV, MKV supported)
4. **Run detection** and monitor progress
5. **Download annotated results**

## 🆘 Need More Help?

If issues persist:
1. Check Docker Desktop is running
2. Ensure sufficient disk space and memory
3. Restart Docker Desktop
4. Try: `docker system prune -a` to clean up
5. Re-run deployment script

The application should now work properly on your MacBook! 🍎