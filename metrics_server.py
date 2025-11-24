"""
Prometheus Metrics Server for ACIS
Exposes metrics on port 8000 for Prometheus scraping
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading
import time

# Inference metrics
inference_counter = Counter(
    'acis_inference_total', 
    'Total number of inferences performed',
    ['model_name', 'device']
)

inference_time_histogram = Histogram(
    'acis_inference_duration_seconds',
    'Time spent on inference',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Detection metrics
detections_counter = Counter(
    'acis_detections_total',
    'Total number of objects detected',
    ['component_type']
)

detection_confidence_histogram = Histogram(
    'acis_detection_confidence',
    'Detection confidence scores',
    ['component_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

low_confidence_counter = Counter(
    'acis_low_confidence_detections_total',
    'Number of detections below confidence threshold',
    ['component_type', 'threshold']
)

# Video processing metrics
video_processing_time = Gauge(
    'acis_video_processing_seconds',
    'Total time to process last video'
)

frames_processed_counter = Counter(
    'acis_frames_processed_total',
    'Total number of frames processed'
)

# SHAP explainability metrics
shap_generation_counter = Counter(
    'acis_shap_explanations_total',
    'Total number of SHAP explanations generated'
)

shap_generation_time_histogram = Histogram(
    'acis_shap_generation_duration_seconds',
    'Time spent generating SHAP explanations',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
# Add these NEW metrics for Customer Feedback and Avg Confidence

# Customer Feedback Metrics
customer_feedback_counter = Counter(
    'acis_customer_feedback_total',
    'Total number of customer feedback submissions',
    ['rating']  # 1-5 star rating
)

customer_feedback_rating_gauge = Gauge(
    'acis_customer_feedback_avg_rating',
    'Average customer feedback rating'
)

# Average Confidence Metric
avg_confidence_gauge = Gauge(
    'acis_avg_confidence_score',
    'Average confidence score across all detections'
)

# Additional useful metrics
confidence_by_component_gauge = Gauge(
    'acis_avg_confidence_by_component',
    'Average confidence score by component type',
    ['component_type']
)
# System metrics
memory_usage_gauge = Gauge(
    'acis_memory_usage_mb',
    'Current memory usage in MB'
)

active_sessions_gauge = Gauge(
    'acis_active_sessions',
    'Number of active user sessions'
)

# Error metrics
error_counter = Counter(
    'acis_errors_total',
    'Total number of errors',
    ['error_type']
)

# Helper functions to record metrics
def record_inference(model_name='yolov8', device='cpu', duration_seconds=0.0):
    """Record an inference event"""
    inference_counter.labels(model_name=model_name, device=device).inc()
    if duration_seconds > 0:
        inference_time_histogram.labels(model_name=model_name).observe(duration_seconds)

        # Helper functions for Customer Feedback
def record_customer_feedback(rating, avg_rating):
    """
    Record customer feedback
    rating: individual rating (1-5)
    avg_rating: calculated average rating from all feedback
    """
    customer_feedback_counter.labels(rating=str(rating)).inc()
    customer_feedback_rating_gauge.set(avg_rating)

def update_avg_confidence(avg_confidence):
    """Update the average confidence score across all detections"""
    avg_confidence_gauge.set(avg_confidence)

def update_confidence_by_component(component_type, avg_confidence):
    """Update average confidence for a specific component type"""
    confidence_by_component_gauge.labels(component_type=component_type).set(avg_confidence)

def update_detection_metrics(component_type, confidence):
    """Update detection metrics"""
    detections_counter.labels(component_type=component_type).inc()
    detection_confidence_histogram.labels(component_type=component_type).observe(confidence)

def record_low_confidence(component_type, threshold):
    """Record low confidence detection"""
    low_confidence_counter.labels(component_type=component_type, threshold=str(threshold)).inc()

def record_detection_error(error_type='unknown'):
    """Record a detection error"""
    error_counter.labels(error_type=error_type).inc()

def record_shap_generation(duration_seconds):
    """Record SHAP explanation generation"""
    shap_generation_counter.inc()
    shap_generation_time_histogram.observe(duration_seconds)

def update_memory_usage(memory_mb):
    """Update memory usage metric"""
    memory_usage_gauge.set(memory_mb)

def update_active_sessions(count):
    """Update active sessions count"""
    active_sessions_gauge.set(count)

# Metrics server management
_metrics_server_started = False
_metrics_server_thread = None

def start_metrics_server(port=8000):
    """Start the Prometheus metrics HTTP server"""
    global _metrics_server_started, _metrics_server_thread
    
    if _metrics_server_started:
        print(f"✅ Metrics server already running on port {port}")
        return
    
    def _run_server():
        try:
            start_http_server(port)
            print(f"✅ Prometheus metrics server started on port {port}")
            print(f"📊 Metrics available at http://localhost:{port}/metrics")
        except Exception as e:
            print(f"⚠️ Failed to start metrics server: {e}")
    
    _metrics_server_thread = threading.Thread(target=_run_server, daemon=True)
    _metrics_server_thread.start()
    _metrics_server_started = True
    
    # Give the server a moment to start
    time.sleep(1)

# Auto-start metrics server when module is imported (in Docker environment)
import os
if os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true':
    # We're in Docker, start the metrics server
    start_metrics_server()
else:
    print("Running in development mode - metrics server not auto-started")