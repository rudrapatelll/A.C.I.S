"""
Prometheus Metrics Server for ACIS
Exposes metrics on port 8000 for Prometheus scraping
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading
import time

# Define metrics
inference_counter = Counter(
    'acis_inference_total', 
    'Total number of inference requests',
    ['model_type']
)

detection_counter = Counter(
    'acis_detections_total',
    'Total number of objects detected',
    ['component_type']
)

inference_duration = Histogram(
    'acis_inference_duration_seconds',
    'Time spent processing inference',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

low_confidence_counter = Counter(
    'acis_low_confidence_detections_total',
    'Number of low confidence detections',
    ['component_type', 'confidence_range']
)

error_counter = Counter(
    'acis_detection_errors_total',
    'Total number of detection errors',
    ['error_type']
)

active_video_processing = Gauge(
    'acis_active_video_processing',
    'Number of videos currently being processed'
)

video_processing_time = Gauge(
    'acis_video_processing_time_seconds',
    'Total time taken to process last video'
)

shap_generation_time = Histogram(
    'acis_shap_generation_seconds',
    'Time spent generating SHAP explanations',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# NEW METRICS
average_confidence = Gauge(
    'acis_average_confidence',
    'Average confidence score across all detections'
)

customer_feedback_rating = Gauge(
    'acis_customer_feedback_rating',
    'Latest customer feedback rating (1-5 stars)'
)

customer_feedback_counter = Counter(
    'acis_customer_feedback_total',
    'Total number of feedback submissions',
    ['rating']
)

# Helper functions to record metrics
def record_inference(model_type='yolo', duration_seconds=0):
    """Record an inference request"""
    inference_counter.labels(model_type=model_type).inc()
    if duration_seconds > 0:
        inference_duration.observe(duration_seconds)

def update_detection_metrics(component_type, confidence, count=1):
    """Update detection metrics"""
    detection_counter.labels(component_type=component_type).inc(count)
    
    # Track low confidence detections
    if confidence < 0.5:
        confidence_range = 'very_low_0-50'
    elif confidence < 0.7:
        confidence_range = 'low_50-70'
    elif confidence < 0.85:
        confidence_range = 'medium_70-85'
    else:
        confidence_range = 'high_85-100'
    
    if confidence < 0.7:
        low_confidence_counter.labels(
            component_type=component_type,
            confidence_range=confidence_range
        ).inc()

def record_low_confidence(component_type, confidence_range):
    """Record low confidence detection"""
    low_confidence_counter.labels(
        component_type=component_type,
        confidence_range=confidence_range
    ).inc()

def record_detection_error(error_type='unknown'):
    """Record a detection error"""
    error_counter.labels(error_type=error_type).inc()

def record_shap_generation(duration_seconds):
    """Record SHAP explanation generation time"""
    shap_generation_time.observe(duration_seconds)

# NEW HELPER FUNCTIONS
def update_average_confidence(confidence_value):
    """Update the average confidence gauge"""
    average_confidence.set(confidence_value)

def record_customer_feedback(rating):
    """Record customer feedback rating"""
    customer_feedback_rating.set(rating)
    customer_feedback_counter.labels(rating=str(rating)).inc()

def start_metrics_server(port=8000):
    """Start the Prometheus metrics HTTP server"""
    try:
        start_http_server(port)
        print(f"✅ Prometheus metrics server started on port {port}")
        print(f"📊 Metrics available at http://localhost:{port}/metrics")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"⚠️  Port {port} already in use - metrics server may already be running")
        else:
            print(f"❌ Error starting metrics server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error starting metrics server: {e}")

def start_metrics_server_thread(port=8000):
    """Start metrics server in a background thread"""
    metrics_thread = threading.Thread(
        target=start_metrics_server,
        args=(port,),
        daemon=True,
        name="MetricsServer"
    )
    metrics_thread.start()
    print(f"🚀 Metrics server thread started")
    return metrics_thread

# Auto-start metrics server when module is imported (for Docker)
if __name__ != "__main__":
    # Give the main app a moment to initialize
    time.sleep(1)
    start_metrics_server_thread()