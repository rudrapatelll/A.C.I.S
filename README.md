# ACIS - Automated Component Identification System

🔍 **ACIS** is a web-based automated component identification system built with Streamlit and powered by YOLOv8 for real-time video component detection.

## 🌟 Features

- **Web-based Interface**: Clean, intuitive Streamlit UI for video upload and analysis
- **Real-time Detection**: YOLOv8-powered component detection with confidence scoring
- **Performance Monitoring**: Comprehensive performance metrics and analytics
- **Multi-format Support**: Supports MP4, AVI, MOV, and MKV video formats
- **Download Results**: Annotated output videos with detection overlays
- **Responsive Design**: Works on desktop and mobile devices
- **Docker Ready**: Easy deployment with Docker and Docker Compose

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- Internet connection for initial setup

### Installation

1. **Clone or download the ACIS project files**

2. **Navigate to the project directory**
   ```bash
   cd acis-streamlit
   ```

3. **Make the deployment script executable**
   ```bash
   chmod +x deploy_acis.sh
   ```

4. **Run the deployment script**
   ```bash
   ./deploy_acis.sh
   ```

5. **Access the application**
   Open your browser and go to: http://localhost:8501

## 📖 Usage Guide

### Step 1: Upload Detection Model
1. Click "Upload Detection Model (.pt)" in the sidebar
2. Select your trained YOLOv8 model file (.pt format)
3. The model will be automatically loaded

### Step 2: Configure Detection Parameters
1. **Confidence Threshold**: Adjust the minimum confidence score (0.1-0.9)
2. Review system information showing device (CPU/CUDA/MPS)

### Step 3: Upload Video
1. Click "Choose a video file" in the main area
2. Select your video file (MP4, AVI, MOV, or MKV)
3. Video information will be displayed

### Step 4: Run Detection
1. Click "🔍 Start Detection" button
2. Monitor progress in real-time
3. View results and download annotated video

### Step 5: Analyze Results
- **Performance Metrics**: View processing speed, memory usage
- **Detection Breakdown**: See component types and counts
- **Download Output**: Get annotated video file

## 🏗️ Architecture

### Components

- **Streamlit Frontend**: Web interface for user interaction
- **ACISDetector**: Core detection engine with YOLOv8
- **PerformanceTracker**: Real-time performance monitoring
- **Docker Container**: Isolated environment with all dependencies

### Technology Stack

- **Backend**: Python 3.10, Streamlit
- **ML Framework**: PyTorch, Ultralytics YOLOv8
- **Computer Vision**: OpenCV
- **Visualization**: Plotly, Pandas
- **Deployment**: Docker, Docker Compose, Nginx

## 📁 Project Structure

```
acis-streamlit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image configuration
├── docker-compose.yml    # Docker Compose setup
├── deploy_acis.sh       # Deployment script
├── nginx.conf           # Nginx configuration for production
├── README.md            # This file
├── uploads/             # Uploaded video files
├── outputs/             # Processed videos with annotations
├── models/              # YOLOv8 model files
└── data/                # Additional data storage
```

## ⚙️ Configuration

### Environment Variables

The application supports the following environment variables:

- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_SERVER_PORT=8501`
- `STREAMLIT_SERVER_ENABLE_CORS=false`
- `STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500` (MB)
- `STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200` (MB)

### Docker Compose Profiles

- **Default**: Basic Streamlit app
- **Production**: Includes Nginx reverse proxy

```bash
# Run with Nginx
docker-compose --profile production up -d
```

## 🔧 Development

### Local Development

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit app**
   ```bash
   streamlit run app.py
   ```

### Adding New Features

The application is modular and extensible:

- **Detector Logic**: Modify `ACISDetector` class in `app.py`
- **UI Components**: Add sections to the Streamlit interface
- **Performance Metrics**: Extend `PerformanceTracker` class
- **Model Support**: Add compatibility for different model formats

## 🐛 Troubleshooting

### Common Issues

**Model Upload Fails**
- Ensure model file is in .pt format
- Check model compatibility with YOLOv8
- Verify file size is under the upload limit

**Video Processing Slow**
- Check system resources (RAM, CPU)
- Adjust confidence threshold for faster processing
- Consider using GPU acceleration if available

**Docker Build Fails**
- Ensure Docker daemon is running
- Check internet connection for downloading dependencies
- Verify sufficient disk space (minimum 2GB)

**Application Not Accessible**
- Check if port 8501 is available
- Verify Docker container is running: `docker-compose ps`
- Review logs: `docker-compose logs acis-app`

### Debug Mode

To run with debug output:
```bash
docker-compose run --rm acis-app streamlit run app.py --logger.level debug
```

## 📊 Performance Optimization

### GPU Acceleration

For better performance with CUDA-compatible GPUs:
1. Ensure NVIDIA Docker runtime is installed
2. Modify Dockerfile to use CUDA-enabled PyTorch
3. Update model loading to use 'cuda' device

### Memory Management

- Large video files (>500MB) may require increased memory
- Monitor container memory usage: `docker stats`
- Adjust Streamlit upload limits if needed

### Batch Processing

For multiple videos:
1. Upload videos individually
2. Process sequentially
3. Download results in batches

## 🔒 Security Considerations

- Application runs in isolated Docker container
- No external network access required
- Uploaded files are contained within the project directory
- Consider enabling authentication for production use

## 📞 Support

For issues and support:

1. **Check the logs**
   ```bash
   docker-compose logs acis-app
   ```

2. **Verify system requirements**
   ```bash
   ./deploy_acis.sh
   ```

3. **Review configuration files**
   - `docker-compose.yml`
   - `requirements.txt`
   - `nginx.conf`

## 🚀 Deployment

### Production Deployment

1. **Build production image**
   ```bash
   docker-compose --profile production build
   ```

2. **Start with reverse proxy**
   ```bash
   docker-compose --profile production up -d
   ```

3. **Configure domain and SSL**
   - Update `nginx.conf` with your domain
   - Add SSL certificate configuration
   - Update firewall rules

### Cloud Deployment

The containerized application can be deployed to:
- **AWS ECS/EKS**: Use the provided Docker image
- **Google Cloud Run**: Build and push to Container Registry
- **Azure Container Instances**: Deploy directly from image
- **DigitalOcean App Platform**: Connect GitHub repository

## 📝 License

This project is based on the original ACIS component detection system.

## 🔄 Version History

- **v1.0.0**: Initial release with Streamlit interface
- **v1.1.0**: Added performance monitoring and analytics
- **v1.2.0**: Improved UI and error handling
- **v2.0.0**: Docker containerization and deployment scripts

---

**Built with ❤️ for automated component identification**