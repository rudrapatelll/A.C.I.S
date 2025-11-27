# 🔍 ACIS - Automated Component Inspection System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered component verification system that automates quality control on automotive assembly lines using Convolutional Neural Networks (CNNs) with YOLOv8, TensorFlow, and OpenCV.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Contents](#repository-contents)
- [System Entry Point](#system-entry-point)
- [Video Demonstration](#video-demonstration)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Deployment Strategy](#deployment-strategy)
- [Monitoring and Metrics](#monitoring-and-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Project Documentation](#project-documentation)
- [Performance Metrics](#performance-metrics)
- [Version Control](#version-control)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

ACIS is designed to eliminate manual errors in automotive assembly lines by instantly confirming the correct part is installed. The system achieves:

- **≥95% accuracy** in component verification
- **200-300ms latency** for real-time processing
- **3-5 images/second throughput** (scalable)
- **≥99% uptime** reliability

This proof-of-concept focuses on 5-6 key dashboard components, establishing a foundation for scaling to 35-40 parts in full production.

### Key Objectives

1. **Automate Quality Control**: Replace manual visual inspection with AI-powered verification
2. **Increase Throughput**: Reduce inspection time per component from seconds to milliseconds
3. **Enhance Accuracy**: Achieve consistent ≥95% detection accuracy under variable conditions
4. **Lower Costs**: Reduce quality control labor and rework expenses

## 📂 Repository Contents

### `src/`
Core system code and application logic:
- **`app.py`**: Main Streamlit web interface with real-time detection
- **`metrics_server.py`**: Prometheus metrics collection and export
- **`requirements.txt`**: Python dependencies (PyTorch, Ultralytics, OpenCV, Streamlit)
- **`models/`**: Trained YOLOv8 detection models (`.pt` files)
- **`utils/`**: Helper functions and preprocessing utilities
- **`data/`**: Sample test videos and validation datasets

### `deployment/`
Containerization and deployment configurations:
- **`Dockerfile`**: Production-ready Docker image with CUDA support
- **`docker-compose.yml`**: Multi-service orchestration (App + Prometheus + Grafana)
- **`deploy_acis.sh`**: Automated deployment script with health checks
- **`check_system.sh`**: System requirements validation script
- **`nginx.conf`**: Reverse proxy configuration for production

### `monitoring/`
Performance tracking and observability:
- **`prometheus.yml`**: Metrics scraping configuration (15s intervals)
- **`grafana/`**: Pre-configured dashboards for real-time monitoring
  - Inference time tracking
  - Detection confidence scores
  - System resource utilization
  - SHAP explanation generation times

### `documentation/`
Comprehensive project documentation:
- **`README.md`**: This file - complete system overview
- **`AIS_Project_Plan_Final.pdf`**: Full project proposal with architecture
- **`TROUBLESHOOTING.md`**: Common deployment issues and solutions
- **`API_DOCUMENTATION.md`**: System endpoints and integration guide

### `videos/`
System demonstrations:
- **`system_demo.mp4`**: Complete walkthrough showing:
  - Model upload and configuration
  - Real-time video processing
  - SHAP explainability visualizations
  - Prometheus metrics collection
  - Grafana dashboard monitoring
  - Feedback collection system

## 🚀 System Entry Point

### Main Application
**Entry point**: `src/app.py`

The Streamlit application provides:
- Web-based video upload interface
- Real-time detection processing with progress tracking
- Configurable confidence thresholds (0.1-0.9)
- SHAP explainability visualizations
- Performance metrics dashboard
- Customer feedback collection

### Running Locally
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r src/requirements.txt

# 3. Place your trained model
cp your_model.pt src/models/best.pt

# 4. Run Streamlit app
streamlit run src/app.py
```

**Access**: http://localhost:8501

### Running with Docker
```bash
# 1. Make deployment script executable
chmod +x deployment/deploy_acis.sh

# 2. Deploy full stack
./deployment/deploy_acis.sh

# 3. Access application
# Main interface: http://localhost:8501
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

## 🎥 Video Demonstration

Watch the complete system demonstration: [`videos/system_demo.mp4`](videos/system_demo.mp4)

### Demo Coverage

1. **System Setup** (0:00-2:00)
   - Docker deployment process
   - Container initialization
   - Health check verification

2. **Detection Workflow** (2:00-5:00)
   - Video upload interface
   - Confidence threshold configuration
   - Real-time processing with progress indicators

3. **SHAP Explainability** (5:00-7:30)
   - Heatmap generation showing model decision regions
   - Multi-object explanation with color-coded bounding boxes
   - Component-specific confidence analysis

4. **Monitoring Backend** (7:30-10:00)
   - Prometheus metrics scraping
   - Real-time metric updates (inference time, confidence, FPS)
   - Grafana dashboard visualization

5. **Results & Feedback** (10:00-12:00)
   - Annotated video download
   - Performance metrics summary
   - Customer feedback submission
   - PDF report generation with SHAP analysis

## ✨ Features

### Core Detection Capabilities
- **YOLOv8n Architecture**: Lightweight 3.2M parameter model optimized for real-time inference
- **Multi-Component Detection**: Simultaneously detects 5-6 dashboard components per frame
- **Confidence Scoring**: Adjustable threshold from 0.1 to 0.9 for precision/recall trade-offs
- **Batch Processing**: Processes every Nth frame (1-3) for speed optimization

### Explainability & Transparency
- **SHAP Integration**: Visual explanations showing which image regions influenced decisions
- **Gradient-Based Saliency**: Heatmaps highlighting critical detection features
- **Multi-Object Explanations**: Color-coded analysis for all detected components
- **Confidence Visualization**: Per-detection confidence scores with historical trends

### Performance Monitoring
- **Real-Time Metrics**: Prometheus-based tracking of:
  - Inference latency (ms/frame)
  - Frames per second (FPS)
  - Memory usage (MB)
  - Detection counts by component type
  - Low-confidence detection rates
  - SHAP generation times
- **Grafana Dashboards**: Pre-configured visualizations with 15-second refresh
- **Health Checks**: Automated container health monitoring with auto-restart

### User Experience
- **Responsive Web Interface**: Clean Streamlit UI with dark/light mode toggle
- **Progress Tracking**: Real-time frame processing indicators
- **Result Download**: Annotated videos with bounding boxes and labels
- **PDF Reports**: Comprehensive performance summaries with SHAP analysis
- **Feedback System**: Star ratings and text feedback with Prometheus integration

## 🛠️ Technology Stack

### Machine Learning
- **PyTorch 2.0+**: Deep learning framework
- **Ultralytics YOLOv8**: Object detection model
- **SHAP**: Model explainability library
- **TensorFlow** (optional): Alternative framework support

### Computer Vision
- **OpenCV 4.8+**: Video processing and annotation
- **Pillow**: Image manipulation
- **NumPy**: Numerical operations

### Web Framework
- **Streamlit 1.28+**: Interactive web interface
- **Plotly**: Interactive performance visualizations
- **Pandas**: Data manipulation and CSV export

### Monitoring & Observability
- **Prometheus**: Time-series metrics collection
- **Grafana**: Dashboard visualization platform
- **prometheus_client**: Python metrics export library

### Deployment
- **Docker 20.10+**: Containerization
- **Docker Compose 2.0+**: Multi-service orchestration
- **Nginx**: Reverse proxy for production
- **Python 3.10**: Base runtime environment

### Cloud Infrastructure (Production)
- **Google Cloud Platform (GCP)**: Cloud provider
- **Vertex AI**: Model training and deployment
- **Cloud Storage**: Blob storage for datasets
- **Cloud SQL**: Structured data storage

## 📦 Deployment Strategy

### Architecture Overview
```
┌─────────────────────────────────────────────────┐
│                Load Balancer                    │
│                  (Nginx)                        │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
┌──────▼──────┐        ┌──────▼──────┐
│   Streamlit  │        │  Prometheus │
│     App      │◄───────┤   Metrics   │
│  (Port 8501) │        │  (Port 9090)│
└──────┬───────┘        └──────┬──────┘
       │                       │
       │                ┌──────▼──────┐
       │                │   Grafana   │
       └────────────────►  Dashboards │
                        │  (Port 3000)│
                        └─────────────┘
```

### Containerization Strategy

**Base Image**: `python:3.10` with full system libraries

**Multi-Stage Build**:
1. **Build Stage**: Install dependencies and compile packages
2. **Production Stage**: Copy only necessary artifacts
3. **Health Check**: Automated `/health` endpoint monitoring

**Resource Allocation**:
- Memory: 4-8GB per container
- CPU: 2-4 cores recommended
- GPU: Optional CUDA support for inference acceleration

### Deployment Methods

#### 1. Local Development
```bash
# Quick start for testing
streamlit run src/app.py
```

#### 2. Docker Single Container
```bash
# Build and run standalone
docker build -t acis:latest -f deployment/Dockerfile .
docker run -p 8501:8501 -v $(pwd)/models:/app/models acis:latest
```

#### 3. Docker Compose (Recommended)
```bash
# Full stack with monitoring
cd deployment
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f acis-app
```

#### 4. Production with Nginx
```bash
# Deploy with reverse proxy
docker-compose --profile production up -d
```

### CI/CD Pipeline (Future)
```yaml
# Planned GitHub Actions workflow
1. Code push → Automated tests
2. Tests pass → Build Docker image
3. Image tagged → Push to registry
4. Deploy to staging → Integration tests
5. Manual approval → Production deployment
```

### Scaling Strategy

**Horizontal Scaling**:
- Deploy multiple app containers behind load balancer
- Shared Prometheus instance for centralized metrics
- Redis for session state management

**Vertical Scaling**:
- Increase container memory for larger models
- Add GPU resources for faster inference
- Enable multi-threading for parallel video processing

## 📊 Monitoring and Metrics

### Prometheus Metrics Collection

**Scraping Configuration**: `monitoring/prometheus.yml`
- **Scrape Interval**: 15 seconds (real-time monitoring)
- **Targets**: 
  - `acis-app:8000` - Application metrics endpoint
  - `prometheus:9090` - Self-monitoring
  - `grafana:3000` - Dashboard health

**Exported Metrics**:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `acis_inference_total` | Counter | Total inference requests by model type |
| `acis_detections_total` | Counter | Objects detected by component type |
| `acis_inference_duration_seconds` | Histogram | Processing time distribution |
| `acis_low_confidence_detections_total` | Counter | Detections below threshold by range |
| `acis_detection_errors_total` | Counter | Failed inferences by error type |
| `acis_active_video_processing` | Gauge | Current videos being processed |
| `acis_video_processing_time_seconds` | Gauge | Last video processing duration |
| `acis_shap_generation_seconds` | Histogram | SHAP explanation generation time |
| `acis_average_confidence` | Gauge | Mean confidence across detections |
| `acis_customer_feedback_rating` | Gauge | Latest user satisfaction rating |
| `acis_customer_feedback_total` | Counter | Feedback submissions by rating |

### Grafana Dashboards

**Access**: http://localhost:3000 (admin/admin)

**Pre-Configured Panels**:

1. **System Overview Dashboard**
   - Current FPS and latency
   - Total detections in last hour
   - Memory and CPU utilization
   - Active video processing gauge

2. **Detection Quality Dashboard**
   - Average confidence trend (15-min window)
   - Low-confidence detection rate
   - Component type distribution (pie chart)
   - Detection error rate over time

3. **Performance Analysis Dashboard**
   - Inference time histogram
   - 95th percentile latency
   - Processing time by video size
   - SHAP generation overhead

4. **User Feedback Dashboard**
   - Average satisfaction rating
   - Feedback volume by hour
   - Rating distribution histogram
   - Recent feedback timeline

### Alerting Rules (Production)
```yaml
# Example alert: High latency
- alert: HighInferenceLatency
  expr: acis_inference_duration_seconds{quantile="0.95"} > 0.3
  for: 5m
  annotations:
    summary: "Inference latency exceeds 300ms threshold"

# Example alert: Low confidence rate
- alert: HighLowConfidenceRate
  expr: rate(acis_low_confidence_detections_total[5m]) > 0.1
  for: 10m
  annotations:
    summary: "More than 10% detections have low confidence"
```

### Monitoring Setup Instructions
```bash
# 1. Deploy monitoring stack
docker-compose up -d prometheus grafana

# 2. Access Prometheus
open http://localhost:9090

# 3. Verify metrics endpoint
curl http://localhost:8000/metrics

# 4. Access Grafana
open http://localhost:3000
# Login: admin / admin

# 5. Add Prometheus data source
# Configuration → Data Sources → Add Prometheus
# URL: http://prometheus:9090

# 6. Import dashboards
# Import from monitoring/grafana/provisioning/
```

## 🔧 Installation

### Prerequisites

- **Operating System**: macOS, Linux, or Windows with WSL2
- **Docker Desktop**: 20.10+ with Docker Compose 2.0+
- **RAM**: Minimum 4GB available (8GB recommended)
- **Disk Space**: 10GB free for Docker images and data
- **Internet**: Required for initial setup

### System Requirements Check
```bash
# Run automated system check
chmod +x deployment/check_system.sh
./deployment/check_system.sh

# Expected output:
# ✅ Docker is installed
# ✅ Docker Compose v2 is installed
# ✅ Available RAM: 8GB
# ✅ Port 8501 is available
```

### Step-by-Step Installation

#### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ACIS-Component-Inspection.git
cd ACIS-Component-Inspection
```

#### 2. Prepare Model File
```bash
# Place your trained YOLOv8 model in src/models/
cp /path/to/your/trained_model.pt src/models/best.pt
```

#### 3. Deploy Application
```bash
# Make deployment script executable
chmod +x deployment/deploy_acis.sh

# Run deployment
./deployment/deploy_acis.sh

# Expected output:
# ✅ Using Docker Compose v2
# 🔧 Building Docker image...
# 🚀 Starting ACIS application...
# ✅ ACIS is running successfully!
```

#### 4. Verify Deployment
```bash
# Check container status
docker-compose ps

# Expected output:
# NAME                STATUS              PORTS
# acis-streamlit      Up (healthy)        0.0.0.0:8501->8501/tcp
# acis-prometheus     Up                  0.0.0.0:9090->9090/tcp
# acis-grafana        Up                  0.0.0.0:3000->3000/tcp

# Check application health
curl http://localhost:8501/_stcore/health

# Expected output: {"status": "ok"}
```

### Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **Main Application** | http://localhost:8501 | None required |
| **Prometheus** | http://localhost:9090 | None required |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Health Check** | http://localhost:8501/_stcore/health | None required |
| **Metrics Endpoint** | http://localhost:8000/metrics | None required |

## 📖 Usage

### 1. Upload and Process Video
```python
# Via Web Interface:
1. Open http://localhost:8501
2. Upload video file (MP4, AVI, MOV, MKV)
3. Configure settings:
   - Processing speed: Fast / Balanced / Ultra Fast
   - Confidence threshold: 0.18 (default) to 0.9
   - Enable SHAP explanations (optional)
4. Click "🔍 Start Detection"
5. Monitor real-time progress
6. Download annotated video and PDF report
```

### 2. Configure Detection Parameters

**Confidence Threshold**:
- **Low (0.1-0.3)**: High recall, more false positives
- **Medium (0.3-0.7)**: Balanced precision/recall
- **High (0.7-0.9)**: High precision, may miss some components

**Processing Speed**:
- **Fast**: Process every 2nd frame (2x speedup)
- **Balanced**: Process every frame (highest accuracy)
- **Ultra Fast**: Process every 3rd frame (3x speedup)

**SHAP Explainability**:
- **Disabled**: Fastest processing
- **Enabled (Single Object)**: Explain primary detection
- **Enabled (All Objects)**: Explain all detections (slowest)
- **Sample Count**: 3-20 frames (default: 5)

### 3. Interpret Results

**Performance Metrics Panel**:
```
┌─────────────────────────────────────────┐
│  Frames: 247/500     Detections: 1,234  │
│  Avg FPS: 43.2       Latency: 23.1 ms   │
└─────────────────────────────────────────┘
```

**Detection Breakdown**:
- Bar chart showing component type distribution
- Total count per component
- Confidence score trends

**SHAP Visualizations**:
- Heatmap overlay on original frame
- Color-coded bounding boxes (#1-#8)
- Per-object confidence scores
- Bounding box coordinates and areas

### 4. Download Results

**Annotated Video**:
- Green bounding boxes with labels
- Confidence scores displayed
- Component names overlaid

**PDF Report Includes**:
- Video metadata (frames, duration, FPS)
- Performance summary table
- Detection breakdown by component
- SHAP explanation samples with images
- Confidence statistics (mean, min, max)

### 5. Provide Feedback
```python
# Customer feedback form:
1. Rate experience: 1-5 stars
2. Provide detailed comments
3. Submit feedback
4. Data stored in outputs/customer_feedback.csv
5. Metrics exported to Prometheus
```

### Example Workflow
```bash
# Complete detection workflow
1. Upload: assembly_line_video.mp4 (500 frames, 30 FPS)
2. Configure: Confidence 0.25, Balanced mode, SHAP enabled (5 samples)
3. Process: ~16 seconds total (2 FPS processing rate)
4. Results:
   - 1,234 total detections across 8 component types
   - 96.3% average confidence
   - 5 SHAP explanations generated
   - 23.1ms average inference time
5. Download: detected_1701234567.mp4 (annotated)
6. Review: ACIS_Report_1701234567.pdf (12 pages)
7. Feedback: 5 stars - "Excellent accuracy!"
```

## 📚 Project Documentation

### Core Documentation

1. **[AIS_Project_Plan_Final.pdf](documentation/AIS_Project_Plan_Final.pdf)**
   - Complete system architecture
   - Hardware/software requirements
   - Trustworthiness and risk management
   - HCI design principles
   - Deployment and maintenance plans

2. **[README.md](README.md)** (this file)
   - Repository overview and navigation
   - Installation and deployment guide
   - Usage instructions and examples
   - Monitoring setup and metrics

3. **[TROUBLESHOOTING.md](documentation/TROUBLESHOOTING.md)**
   - Common deployment errors
   - Docker Compose compatibility issues
   - ARM64/Apple Silicon fixes
   - Port conflict resolution
   - Memory and disk space problems

4. **[API_DOCUMENTATION.md](documentation/API_DOCUMENTATION.md)**
   - Streamlit interface endpoints
   - Prometheus metrics API
   - Model inference API
   - SHAP explanation generation

### Key Project Sections

**Problem Definition** (Project Plan §1):
- Automotive assembly quality control challenges
- Manual inspection limitations
- Stakeholder requirements (QA, operators, engineers)

**Data Management** (Project Plan §5):
- Proprietary dataset collection (2,000 images, 8 component types)
- K-anonymity for worker privacy
- Augmentation strategy (rotation, brightness, zoom)
- Versioning with Git and CSV metadata

**Model Development** (Project Plan §6):
- YOLOv8n architecture (3.2M parameters)
- Training on Google Colab (NVIDIA A100 GPU)
- 99.5% mAP@0.5 across all component classes
- 0.3ms inference time (1000x faster than target)

**Deployment Strategy** (Project Plan §7):
- Docker containerization with Compose
- Prometheus + Grafana monitoring stack
- Nginx reverse proxy for production
- Phased rollout (shadow → assisted → autonomous)

**Monitoring & Maintenance** (Project Plan §8):
- Weekly confidence trend reviews
- Monthly bias audits with Fairlearn
- Quarterly SHAP explainability assessments
- Automated retraining on 2% accuracy drop

**HCI Design** (Project Plan §4):
- Primary persona: Jordan Alvarez (assembly operator)
- Secondary persona: Priya Desai (QA manager)
- Figma wireframes for 4 station states
- WCAG 2.2 Level AA accessibility compliance

## 🎯 Performance Metrics

### Model Performance (Validation Set)

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| mAP@0.5 | ≥95% | 99.5% | ✅ Exceeded |
| mAP@0.5-0.95 | - | 97.5% | ✅ Excellent |
| Precision | ≥93% | 99.5% | ✅ Exceeded |
| Recall | ≥96% | 99.4% | ✅ Exceeded |
| Inference Time | ≤300ms | 0.3ms | ✅ 1000x faster |

### Per-Component Performance

| Component | mAP@0.5 | Precision | Recall | Count |
|-----------|---------|-----------|--------|-------|
| Bezel | 0.995 | 1.000 | 1.000 | 99 |
| Screen | 0.995 | 0.999 | 1.000 | 99 |
| ac_ctr_left | 0.995 | 0.989 | 0.999 | 373 |
| ac_ctr_right | 0.995 | 0.994 | 0.988 | 248 |
| usb_aux | 0.990 | 0.989 | 0.985 | 245 |
| Wiper | 0.984 | 0.989 | 0.977 | 99 |
| fdr_ip | 0.974 | 0.999 | 0.985 | 534 |
| Lights | 0.995 | 0.994 | 0.979 | 99 |

### System Performance (Production)

| Metric | Value | Notes |
|--------|-------|-------|
| Training Time | 13 minutes | NVIDIA A100-SXM4-80GB GPU |
| Model Size | 6.2 MB | Lightweight for edge deployment |
| Memory Usage | 12.3 GB | Per container |
| Uptime | 99.2% | 48-hour test period |
| Error Rate | 0.0% | Zero crashes during testing |

### Explainability Performance

| Metric | Value | Impact |
|--------|-------|--------|
| SHAP Generation Time | 2.3s avg | Per frame explanation |
| Samples Generated | 5-20 | Configurable |
| Avg Confidence (Sampled) | 93-96% | Representative of overall |
| Visualization Overhead | +15% | Total processing time |

## 🔄 Version Control

### Branching Strategy
```
main (production-ready)
  ├── develop (integration branch)
  │   ├── feature/shap-explainability
  │   ├── feature/prometheus-metrics
  │   ├── feature/grafana-dashboards
  │   └── feature/customer-feedback
  └── hotfix/docker-compose-v2
```

### Commit Convention
```bash
# Format: <type>(<scope>): <subject>

feat(detection): Add SHAP explainability visualizations
fix(deployment): Update Docker Compose syntax for v2
docs(readme): Add monitoring setup instructions
perf(inference): Optimize preprocessing pipeline
test(metrics): Add Prometheus metrics unit tests
```

### Team Collaboration (Solo Project)

**Development Workflow**:
1. Create feature branch from `develop`
2. Implement feature with incremental commits
3. Test locally with `pytest` and manual QA
4. Update documentation (README, comments)
5. Merge to `develop` via pull request
6. Deploy to staging for integration testing
7. Merge to `main` for production release

**Code Review Checklist**:
- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Unit tests pass with >80% coverage
- [ ] Docker builds successfully
- [ ] Metrics are properly exported
- [ ] Documentation is updated
- [ ] No sensitive data in commits

### Release Management

**Version Numbering**: Semantic Versioning (MAJOR.MINOR.PATCH)
```
v1.0.0 - Initial release (YOLOv8 detection)
v1.1.0 - Added Prometheus metrics
v1.2.0 - SHAP explainability feature
v1.3.0 - Customer feedback system
v2.0.0 - Docker Compose refactor
```

**Git Tags**:
```bash
git tag -a v2.0.0 -m "Production-ready with monitoring stack"
git push origin v2.0.0
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Docker Compose Command Not Found

**Error**:
```bash
zsh: command not found: docker-compose
```

**Solution**:
```bash
# Modern Docker includes Compose v2
docker compose version

# Update deploy script to use new syntax
sed -i 's/docker-compose/docker compose/g' deployment/deploy_acis.sh
```

#### 2. Package Installation Fails (ARM64)

**Error**:
```
E: Package 'libgl1-mesa-glx' has no installation candidate
```

**Solution**:
```dockerfile
# Use full Python image in Dockerfile
FROM python:3.10  # Instead of python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*
```

#### 3. Port Already in Use

**Error**:
```
Error starting userland proxy: listen tcp4 0.0.0.0:8501: bind: address already in use
```

**Solution**:
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8502:8501"  # Use different host port
```

#### 4. Model File Not Found

**Error**:
```
FileNotFoundError: Model file 'models/best.pt' not found
```

**Solution**:
```bash
# Check model file exists
ls -lh src/models/best.pt

# If missing, download or copy model
cp /path/to/trained_model.pt src/models/best.pt

# Verify Docker volume mount
docker-compose exec acis-app ls -lh /app/models/
```

#### 5. Out of Memory (OOM)

**Error**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution**:
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 8GB

# Or reduce batch size in code
batch
For more issues, see [TROUBLESHOOTING.md](documentation/TROUBLESHOOTING.md).

## 🤝 Contributing

This is a student project completed as part of an AI Systems course. While contributions are welcome for educational purposes, please note:

1. **Fork the repository** before making changes
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request** with detailed description

### Code Style

- Follow [PEP 8](https://pep8.org/) for Python code
- Use [Black](https://github.com/psf/black) for formatting
- Add docstrings to all functions
- Include type hints where applicable

### Testing
```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Lint code
flake8 src/
black --check src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 object detection framework
- **Streamlit**: Rapid web application development
- **Prometheus & Grafana**: Monitoring and observability
- **Google Cloud Platform**: GPU compute for model training
- **University of Florida**: Course support and guidance

## 📧 Contact

**Rudra Patel**
- Email: patel.rudra@ufl.edu
- Student ID: 20034606
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Built with ❤️ for automated component inspection**