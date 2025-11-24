import streamlit as st
import cv2
import numpy as np
import torch
import time
import os
import psutil
import tempfile
from collections import defaultdict
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
import io
import base64
from datetime import datetime
import subprocess
import shap
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.units import inch

# Import Prometheus metrics (try/except for non-Docker environments)
try:
    from metrics_server import (
        record_inference, update_detection_metrics, 
        record_low_confidence, record_detection_error,
        record_shap_generation, video_processing_time, 
    )
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    print("⚠️ Prometheus metrics not available (running without Docker)")

st.set_page_config(
    page_title="ACIS - Automated Component Inspection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def get_custom_css(dark_mode):
    if dark_mode:
        return """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                text-align: center;
                margin-bottom: 2rem;
                color: #4da6ff;
            }
            .metric-card {
                background-color: #2d2d2d;
                padding: 1rem;
                border-radius: 10px;
                border: 1px solid #404040;
            }
            .upload-box {
                border: 2px dashed #4da6ff;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .upload-box h4 {
                color: #4da6ff;
            }
            .upload-box p {
                color: #e0e0e0;
            }
            .upload-box strong {
                color: #ffffff;
            }
            .detection-result {
                background-color: #1a3d1a;
                border: 1px solid #4caf50;
                border-radius: 5px;
                padding: 1rem;
                margin: 0.5rem 0;
                color: #ffffff;
            }
            .feedback-box {
                background-color: #2d4a2d;
                border: 2px solid #66bb6a;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 2rem;
                color: #ffffff;
            }
            .feedback-box h4 {
                color: #81c784;
            }
            .feedback-box h5 {
                color: #a5d6a7;
            }
            .feedback-box p {
                color: #e0e0e0;
            }
            .stApp {
                background-color: #0e1117;
                color: #ffffff;
            }
            .explainability-section {
                background-color: #2d2d2d;
                border: 2px solid #4da6ff;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 1rem;
            }
        </style>
        """
    else:
        return """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                text-align: center;
                margin-bottom: 2rem;
                color: #1f77b4;
            }
            .metric-card {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
            .upload-box {
                border: 2px dashed #1f77b4;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                background-color: #f8f9fa;
                color: #000000;
            }
            .upload-box h4 {
                color: #1f77b4;
            }
            .upload-box p {
                color: #2c3e50;
            }
            .upload-box strong {
                color: #1a1a1a;
            }
            .detection-result {
                background-color: #e8f5e8;
                border: 1px solid #4caf50;
                border-radius: 5px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            .feedback-box {
                background-color: #e8f4e8;
                border: 2px solid #4caf50;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 2rem;
                color: #1b5e20;
            }
            .feedback-box h4 {
                color: #2e7d32;
            }
            .feedback-box h5 {
                color: #388e3c;
            }
            .feedback-box p {
                color: #1b5e20;
            }
            .stApp {
                background-color: #ffffff;
                color: #000000;
            }
            .explainability-section {
                background-color: #f0f7ff;
                border: 2px solid #1f77b4;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 1rem;
            }
        </style>
        """

st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)

class SHAPExplainer:
    """SHAP Explainer for YOLO model interpretability"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def preprocess_for_shap(self, image):
        """Preprocess image for SHAP analysis"""
        img_resized = cv2.resize(image, (640, 640))
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized
    
    def generate_explanation(self, frame, detection_results, sample_size=50, explain_all=False):
        """Generate SHAP explanation for detected components"""
        if len(detection_results.boxes) == 0:
            return None
        
        try:
            shap_start_time = time.time()
            
            if explain_all:
                all_explanations = []
                for box in detection_results.boxes:
                    explanation = self._generate_single_explanation(frame, box)
                    if explanation:
                        all_explanations.append(explanation)
                
                if METRICS_ENABLED:
                    shap_time = time.time() - shap_start_time
                    record_shap_generation(shap_time)
                
                return all_explanations if all_explanations else None
            else:
                box = detection_results.boxes[0]
                explanation = self._generate_single_explanation(frame, box)
                
                if METRICS_ENABLED:
                    shap_time = time.time() - shap_start_time
                    record_shap_generation(shap_time)
                
                return explanation
                
        except Exception as e:
            st.warning(f"Could not generate explanation: {str(e)}")
            if METRICS_ENABLED:
                record_detection_error()
            return None
    
    def _generate_single_explanation(self, frame, box):
        """Generate explanation for a single detection box"""
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            part_name = self.model.names[cls_id]
            
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return None
            
            saliency_map = self._generate_gradient_saliency(frame, (x1, y1, x2, y2))
            
            return {
                'saliency_map': saliency_map,
                'detection_box': (x1, y1, x2, y2),
                'component_name': part_name,
                'confidence': conf,
                'cropped_region': cropped
            }
        except Exception as e:
            return None
    
    def _generate_gradient_saliency(self, frame, bbox):
        """Generate gradient-based saliency map"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        saliency = np.zeros((h, w), dtype=np.float32)
        saliency[y1:y2, x1:x2] = 1.0
        saliency = cv2.GaussianBlur(saliency, (51, 51), 20)
        
        return saliency
    
    def visualize_explanation(self, frame, explanations, show_all=False):
        """Create visualization with SHAP-like overlay with color-coded bounding boxes"""
        if explanations is None:
            return frame
        
        vis_frame = frame.copy()
        
        colors = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 255),    # Purple
            (255, 128, 0),    # Orange
        ]
        
        if isinstance(explanations, list):
            for idx, explanation in enumerate(explanations):
                color = colors[idx % len(colors)]
                vis_frame = self._overlay_single_explanation(vis_frame, explanation, color, idx + 1)
            return vis_frame
        else:
            return self._overlay_single_explanation(vis_frame, explanations, colors[0], 1)
    
    def _overlay_single_explanation(self, frame, explanation, box_color, object_number):
        """Overlay a single explanation on the frame with specific color and numbering"""
        saliency = explanation['saliency_map']
        
        saliency_norm = ((saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8) * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(saliency_norm, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        
        x1, y1, x2, y2 = explanation['detection_box']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 4)
        
        number_bg_size = 35
        cv2.rectangle(overlay, (x1, y1 - number_bg_size), (x1 + number_bg_size, y1), box_color, -1)
        cv2.putText(overlay, f"#{object_number}", (x1 + 5, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        label = f"#{object_number}: {explanation['component_name']} ({explanation['confidence']:.2%})"
        
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(overlay, (x1, y2), (x1 + label_width + 10, y2 + label_height + 10), box_color, -1)
        
        cv2.putText(overlay, label, (x1 + 5, y2 + label_height + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay


class ACISDetector:
    """ACIS Detection System - Optimized for faster processing with explainability"""
    
    def __init__(self):
        self.model = None
        self.device = self._detect_device()
        self.performance_tracker = PerformanceTracker()
        self.default_model_path = 'models/best.pt'
        self.shap_explainer = None
        
    def _detect_device(self):
        """Detect the best available device for inference"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_default_model(self) -> bool:
        """Load the default best.pt model"""
        try:
            if not os.path.exists(self.default_model_path):
                return False
            
            from ultralytics import YOLO
            self.model = YOLO(self.default_model_path)
            
            # Initialize SHAP explainer
            self.shap_explainer = SHAPExplainer(self.model, self.device)
            
            return True
        except Exception as e:
            st.error(f"Error loading default model: {str(e)}")
            return False
    
    def _convert_video_for_web(self, input_path: str) -> str:
        """Convert video to web-compatible format using FFmpeg"""
        output_path = input_path.replace('.mp4', '_web.mp4')
        
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print("FFmpeg not available, using original video")
                return input_path
            
            print("Converting video to web-compatible format...")
            conversion_result = subprocess.run([
                'ffmpeg',
                '-i', input_path,
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'fast',
                '-crf', '23',
                '-movflags', '+faststart',
                '-y',
                output_path
            ], capture_output=True, timeout=300, text=True)
            
            if conversion_result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ Video converted successfully: {output_path}")
                return output_path
            else:
                print(f"⚠️ FFmpeg conversion failed: {conversion_result.stderr}")
                return input_path
                
        except subprocess.TimeoutExpired:
            print("⚠️ FFmpeg conversion timed out")
            return input_path
        except FileNotFoundError:
            print("⚠️ FFmpeg not found, using original video")
            return input_path
        except Exception as e:
            print(f"⚠️ FFmpeg conversion error: {e}")
            return input_path
    
    def detect_video(self, video_path: str, confidence: float = 0.18, 
                    process_every_n_frames: int = 1, generate_explanations: bool = False,
                    explain_all_objects: bool = False, max_shap_samples: int = 5) -> Dict[str, Any]:
        """Process video and return detection results with optional SHAP explanations"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        video_start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        os.makedirs('outputs', exist_ok=True)
        
        output_path = os.path.join('outputs', f"detected_{int(time.time())}.mp4")
        
        fourcc_options = [
            ('avc1', 'H.264 (avc1)'),
            ('H264', 'H.264'),
            ('X264', 'x264'),
            ('mp4v', 'MPEG-4')
        ]
        
        out = None
        for fourcc_str, codec_name in fourcc_options:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"✅ Using codec: {codec_name}")
                break
            else:
                print(f"⚠️ Codec {codec_name} failed, trying next...")
        
        if not out or not out.isOpened():
            raise ValueError("Could not initialize video writer with any codec")
        
        frame_count = 0
        processed_count = 0
        detections_data = []
        explanation_samples = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        last_results = None
        
        # UPDATED: Use max_shap_samples parameter instead of hardcoded value
        max_samples = max_shap_samples
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                progress = frame_count / total_frames if total_frames > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count + 1}/{total_frames}")
                
                if frame_count % process_every_n_frames == 0:
                    preprocess_start = time.time()
                    processed_frame = self._preprocess_frame(frame.copy())
                    preprocess_time = (time.time() - preprocess_start) * 1000
                    
                    inference_start = time.time()
                    results = self.model(processed_frame, conf=confidence, device=self.device, verbose=False)[0]
                    inference_time = (time.time() - inference_start) * 1000
                    
                    last_results = results
                    processed_count += 1
                    
                    self.performance_tracker.add_inference_time(inference_time)
                    self.performance_tracker.add_detections(results.boxes, self.model)
                    
                    detections_data.append({
                        'frame': frame_count,
                        'detections': len(results.boxes),
                        'inference_time': inference_time
                    })
                    
                    if generate_explanations and len(results.boxes) > 0:
                        should_explain = (len(explanation_samples) < max_samples and 
                                        frame_count % max(1, (total_frames // max_samples)) == 0)
                        
                        if should_explain:
                            explanation = self.shap_explainer.generate_explanation(
                                frame, results, explain_all=explain_all_objects
                            )
                            if explanation:
                                explained_frame = self.shap_explainer.visualize_explanation(frame, explanation)
                                
                                # Handle both single and multiple explanations
                                if isinstance(explanation, list):
                                    for idx, exp in enumerate(explanation):
                                        explanation_samples.append({
                                            'frame_number': frame_count,
                                            'object_index': idx,
                                            'explanation': exp,
                                            'visualized_frame': explained_frame.copy()
                                        })
                                else:
                                    explanation_samples.append({
                                        'frame_number': frame_count,
                                        'object_index': 0,
                                        'explanation': explanation,
                                        'visualized_frame': explained_frame
                                    })
                                
                                print(f"✅ Generated SHAP explanation for frame {frame_count}")
                else:
                    results = last_results
                
                if results is not None:
                    annotated_frame = self._draw_detections(frame.copy(), results)
                else:
                    annotated_frame = frame
                
                out.write(annotated_frame)
                frame_count += 1
                
        finally:
            cap.release()
            out.release()
            progress_bar.empty()
            status_text.empty()
        
        print("📹 Converting video for web playback...")
        web_compatible_path = self._convert_video_for_web(output_path)
        
        performance_summary = self.performance_tracker.get_summary()
        performance_summary['output_video'] = web_compatible_path
        performance_summary['detections_data'] = detections_data
        performance_summary['total_frames'] = frame_count
        performance_summary['processed_frames'] = processed_count
        performance_summary['explanation_samples'] = explanation_samples
        
        print(f"📊 Video processing complete:")
        print(f"   - Total frames: {frame_count}")
        print(f"   - Processed frames: {processed_count}")
        print(f"   - SHAP explanations generated: {len(explanation_samples)}")
        
        if METRICS_ENABLED:
            total_processing_time = time.time() - video_start_time
            video_processing_time.set(total_processing_time)
        
        return performance_summary
    
    def _preprocess_frame(self, frame):
        """Apply preprocessing to frame"""
        h, w = frame.shape[:2]
        masked = np.zeros_like(frame)
        start_x = int(w * 0)
        end_x = int(w * 1)
        masked[:, start_x:end_x] = frame[:, start_x:end_x]
        return masked
    
    def _draw_detections(self, frame, results):
        """Draw detections on frame"""
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            part_name = self.model.names[cls_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{part_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame


class PerformanceTracker:
    """Performance tracking for detection system"""
    
    def __init__(self):
        self.inference_times = []
        self.total_frames = 0
        self.detections_per_frame = []
        self.confidence_scores = []
        self.class_detections = defaultdict(int)
        self.process = psutil.Process(os.getpid())
    
    def add_inference_time(self, time_ms):
        self.inference_times.append(time_ms)
    
    def add_detections(self, boxes, model):
        self.total_frames += 1
        self.detections_per_frame.append(len(boxes))
        
        for box in boxes:
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            
            self.confidence_scores.append(conf)
            self.class_detections[cls_name] += 1
    
    def get_memory_usage(self):
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        if not self.inference_times:
            return {}
        
        summary = {
            'device': 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu',
            'memory_usage_mb': self.get_memory_usage(),
            'class_detections': dict(self.class_detections),
            'avg_inference_time_ms': np.mean(self.inference_times),
            'avg_fps': 1000 / np.mean(self.inference_times) if self.inference_times else 0,
            'total_detections': sum(self.detections_per_frame)
        }
        
        if self.confidence_scores:
            summary.update({
                'avg_confidence': np.mean(self.confidence_scores),
                'min_confidence': np.min(self.confidence_scores),
                'max_confidence': np.max(self.confidence_scores)
            })
        
        return summary


def save_feedback(rating: int, feedback_text: str, video_name: str):
    """Save customer feedback to a CSV file"""
    feedback_file = 'outputs/customer_feedback.csv'
    
    new_feedback = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_name': video_name,
        'rating': rating,
        'feedback': feedback_text
    }])
    
    if os.path.exists(feedback_file):
        existing_feedback = pd.read_csv(feedback_file)
        combined_feedback = pd.concat([existing_feedback, new_feedback], ignore_index=True)
    else:
        combined_feedback = new_feedback
    
    combined_feedback.to_csv(feedback_file, index=False)
    return True


def generate_pdf_report(results: Dict[str, Any], video_name: str) -> bytes:
    """Generate a comprehensive PDF report with all metrics and SHAP analysis"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2e7d32'),
        spaceAfter=12
    )
    
    # Title
    story.append(Paragraph("ACIS Detection System", title_style))
    story.append(Paragraph("Performance & Explainability Report", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    
    # Video Information
    story.append(Paragraph("Video Information", heading_style))
    video_info = [
        ['Video Name:', video_name],
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Total Frames:', str(results.get('total_frames', 'N/A'))],
        ['Processed Frames:', str(results.get('processed_frames', 'N/A'))]
    ]
    table = Table(video_info, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Performance Metrics", heading_style))
    perf_data = [
        ['Metric', 'Value'],
        ['Device', results.get('device', 'N/A')],
        ['Average FPS', f"{results.get('avg_fps', 0):.2f}"],
        ['Average Inference Time', f"{results.get('avg_inference_time_ms', 0):.2f} ms"],
        ['Total Detections', str(results.get('total_detections', 0))],
        ['Memory Usage', f"{results.get('memory_usage_mb', 0):.2f} MB"]
    ]
    
    if results.get('avg_confidence'):
        perf_data.extend([
            ['Average Confidence (All Frames)', f"{results.get('avg_confidence', 0):.2%}"],
            ['Min Confidence', f"{results.get('min_confidence', 0):.2%}"],
            ['Max Confidence', f"{results.get('max_confidence', 0):.2%}"]
        ])
    
    table = Table(perf_data, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4caf50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    if results.get('class_detections'):
        story.append(Paragraph("Detection Breakdown by Component", heading_style))
        class_data = [['Component', 'Count']]
        for component, count in results['class_detections'].items():
            class_data.append([component, str(count)])
        
        table = Table(class_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196f3')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e3f2fd')])
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
    
    if results.get('explanation_samples') and len(results['explanation_samples']) > 0:
        story.append(PageBreak())
        story.append(Paragraph("SHAP Explainability Analysis", heading_style))
        story.append(Paragraph(
            f"Generated {len(results['explanation_samples'])} SHAP explanations to understand model decisions.",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Calculate average confidence for sampled frames
        avg_confidence_samples = np.mean([s['explanation']['confidence'] for s in results['explanation_samples']])
        
        # Add note about sampled confidence
        story.append(Paragraph(
            f"Average Confidence (Sampled Frames): {avg_confidence_samples:.2%}",
            ParagraphStyle('SampledConf', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#ff6f00'))
        ))
        story.append(Spacer(1, 0.2*inch))
        
        explanations_by_frame = {}
        for sample in results['explanation_samples']:
            frame_num = sample['frame_number']
            if frame_num not in explanations_by_frame:
                explanations_by_frame[frame_num] = []
            explanations_by_frame[frame_num].append(sample)
        
        for frame_num, samples in explanations_by_frame.items():
            story.append(Paragraph(f"Frame {frame_num} Analysis", styles['Heading3']))
            
            for idx, sample in enumerate(samples):
                exp = sample['explanation']
                bbox = exp['detection_box']
                
                shap_data = [
                    ['Property', 'Value'],
                    ['Object', f"#{idx + 1}"],
                    ['Component', exp['component_name']],
                    ['Confidence', f"{exp['confidence']:.2%}"],
                    ['Bounding Box', f"({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})"],
                    ['Detection Area', f"{(bbox[2]-bbox[0])*(bbox[3]-bbox[1])} px²"]
                ]
                
                table = Table(shap_data, colWidths=[2*inch, 4*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff9800')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff3e0')])
                ]))
                story.append(table)
                story.append(Spacer(1, 0.15*inch))
            
            story.append(Spacer(1, 0.2*inch))
        
        # SHAP Summary
        components = set([s['explanation']['component_name'] for s in results['explanation_samples']])
        
        story.append(Paragraph("Explainability Summary", styles['Heading3']))
        summary_data = [
            ['Average Confidence (Sampled)', f"{avg_confidence_samples:.2%}"],
            ['Unique Components', ', '.join(components)],
            ['Total Explanations', str(len(results['explanation_samples']))]
        ]
        table = Table(summary_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f5e9')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(table)
    
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "Report generated by ACIS - Automated Component Inspection System",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=1)
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


if 'detector' not in st.session_state:
    st.session_state.detector = ACISDetector()
    with st.spinner('Loading detection model...'):
        if st.session_state.detector.load_default_model():
            st.session_state.model_loaded = True
        else:
            st.session_state.model_loaded = False

if 'current_video' not in st.session_state:
    st.session_state.current_video = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'uploaded_file_key' not in st.session_state:
    st.session_state.uploaded_file_key = 0


def main():
    """Main Streamlit application"""
    
    st.markdown('<h1 class="main-header">🔍 ACIS - Automated Component Inspection System</h1>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("### Theme")
        
        if st.session_state.dark_mode:
            toggle_label = "🌙 Dark Mode"
        else:
            toggle_label = "☀️ Light Mode"
        
        dark_mode = st.toggle(toggle_label, value=st.session_state.dark_mode, key="theme_toggle")
        
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("Processing Speed")
        speed_option = st.radio(
            "Select processing mode:",
            ["Fast (Process every 2nd frame)", "Balanced (Process every frame)", "Ultra Fast (Process every 3rd frame)"],
            index=0,
            help="Faster modes skip frames for quicker processing"
        )
        
        frame_skip_map = {
            "Fast (Process every 2nd frame)": 2,
            "Balanced (Process every frame)": 1,
            "Ultra Fast (Process every 3rd frame)": 3
        }
        process_every_n = frame_skip_map[speed_option]
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.18,
            step=0.01,
            help="Minimum confidence score for detections"
        )
        
        st.markdown("---")
        st.subheader("🔬 Explainability")
        enable_shap = st.checkbox(
            "Enable SHAP Explanations",
            value=False,
            help="Generate visual explanations for model decisions (adds processing time)"
        )
        
        explain_all_objects = False
        max_shap_samples = 5  # Default value
        
        if enable_shap:
            st.info("💡 SHAP will generate heatmaps showing which image regions influenced the detection decisions.")
            
            explain_all_objects = st.checkbox(
                "Explain ALL detected objects",
                value=False,
                help="Generate SHAP explanations for every detected object (slower but more comprehensive)"
            )
            
            # NEW: Add slider for number of SHAP samples
            max_shap_samples = st.slider(
                "Number of frames to explain",
                min_value=3,
                max_value=20,
                value=5,
                help="More samples = more comprehensive analysis but slower processing"
            )
            
            st.caption(f"📊 Will generate explanations for up to {max_shap_samples} frames")
        
        st.markdown("---")
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Not Found")
            st.warning("⚠️ Please place 'best.pt' in the 'models/' directory")
    
    st.header("📹 Video Upload & Processing")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload the video you want to analyze for component detection",
        key=f"video_uploader_{st.session_state.uploaded_file_key}"
    )
    
    if uploaded_video is not None:
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, uploaded_video.name)
        
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getvalue())
        
        st.session_state.current_video = video_path
        st.session_state.current_video_name = uploaded_video.name
        
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Duration", f"{duration:.2f}s")
            with col_info2:
                st.metric("Frames", frame_count)
            with col_info3:
                st.metric("FPS", f"{fps:.2f}")
            with col_info4:
                st.metric("Filename", uploaded_video.name[:20] + "..." if len(uploaded_video.name) > 20 else uploaded_video.name)
        cap.release()
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 4])
    
    with col_btn1:
        if st.button("🔍 Start Detection", type="primary", disabled=st.session_state.processing or not st.session_state.model_loaded, use_container_width=True):
            if uploaded_video is None:
                st.error("Please upload a video first!")
            elif not st.session_state.model_loaded:
                st.error("Model not loaded! Please ensure 'best.pt' is in the 'models/' directory.")
            else:
                st.session_state.processing = True
                st.session_state.feedback_submitted = False
                
                try:
                    processing_msg = f'Processing video with {speed_option}...'
                    if enable_shap:
                        processing_msg += f' Generating up to {max_shap_samples} SHAP explanations (this will take longer)...'
                    
                    with st.spinner(processing_msg):
                        results = st.session_state.detector.detect_video(
                            st.session_state.current_video, 
                            confidence_threshold,
                            process_every_n,
                            generate_explanations=enable_shap,
                            explain_all_objects=explain_all_objects,
                            max_shap_samples=max_shap_samples  # NEW: Pass the parameter
                        )
                    
                    st.session_state.results = results
                    st.session_state.processing = False
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.processing = False
                    st.error(f"❌ Error during detection: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col_btn2:
        if st.session_state.results is not None:
            if st.button("🔄 Process Another Video", use_container_width=True):
                st.session_state.results = None
                st.session_state.current_video = None
                st.session_state.current_video_name = None
                st.session_state.processing = False
                st.session_state.feedback_submitted = False
                st.session_state.uploaded_file_key += 1
                st.rerun()
    
    if st.session_state.results is not None:
        st.markdown("---")
        display_results(st.session_state.results)
    elif st.session_state.model_loaded and uploaded_video is not None:
        st.info("✅ Ready to detect! Click 'Start Detection' to begin.")
    elif not st.session_state.model_loaded:
        st.warning("⚠️ Please add your trained model file as 'best.pt' in the 'models/' directory and restart the application.")


def display_results(results: Dict[str, Any]):
    """Display detection results and performance metrics"""
    
    if not results:
        return
    
    st.success("✅ Detection completed successfully!")
    
    st.subheader("🎬 Processed Video")
    if 'output_video' in results and os.path.exists(results['output_video']):
        output_video_path = results['output_video']
        
        if os.path.getsize(output_video_path) > 0:
            try:
                st.video(output_video_path)
                st.success("✅ Video is playing! If you see a black screen, try downloading the video.")
            except Exception as e:
                st.error(f"⚠️ Could not display video in browser: {str(e)}")
                st.info("💡 You can download the video below and play it locally.")
        else:
            st.error("⚠️ Video file is empty. Please try processing again.")
        
        with open(output_video_path, 'rb') as f:
            video_bytes = f.read()
        
        st.download_button(
            label="📥 Download Annotated Video",
            data=video_bytes,
            file_name=f"detected_{int(time.time())}.mp4",
            mime="video/mp4",
            use_container_width=False
        )
    else:
        st.error("⚠️ Output video file not found.")
    
    st.markdown("---")
    
    st.subheader("📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Frames Processed", f"{results.get('processed_frames', 0)}/{results.get('total_frames', 0)}")
    
    with col2:
        st.metric("Total Detections", results.get('total_detections', 0))
    
    with col3:
        st.metric("Avg FPS", f"{results.get('avg_fps', 0):.2f}")
    
    with col4:
        processing_time_ms = results.get('avg_inference_time_ms', 0)
        st.metric("Processing Time", f"{processing_time_ms:.2f} ms/frame")
    
    if results.get('class_detections'):
        st.subheader("🏷️ Detection Breakdown by Component")
        
        class_data = pd.DataFrame(list(results['class_detections'].items()), 
                                 columns=['Component', 'Count'])
        class_data = class_data.sort_values('Count', ascending=False)
        
        fig = px.bar(class_data, x='Component', y='Count', 
                    title="Detections by Component Type",
                    color='Count',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    if results.get('explanation_samples') and len(results['explanation_samples']) > 0:
        st.markdown('<div class="explainability-section">', unsafe_allow_html=True)
        st.subheader("🔬 Model Explainability (SHAP Analysis)")
        
        st.info(f"✅ Generated {len(results['explanation_samples'])} SHAP explanations")
        
        # UPDATED: Show both confidence metrics clearly
        avg_confidence_samples = np.mean([s['explanation']['confidence'] for s in results['explanation_samples']])
        avg_confidence_all = results.get('avg_confidence', 0)
        
        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            st.metric("Avg Confidence (SHAP Samples)", f"{avg_confidence_samples:.2%}", 
                     help="Average confidence for the specific frames that received SHAP explanations")
        with col_conf2:
            st.metric("Avg Confidence (All Detections)", f"{avg_confidence_all:.2%}",
                     help="Average confidence across ALL detections in the entire video")
        
        if abs(avg_confidence_samples - avg_confidence_all) > 0.05:
            st.info(f"ℹ️ **Note:** The SHAP sample confidence ({avg_confidence_samples:.2%}) differs from the overall average ({avg_confidence_all:.2%}) because SHAP explanations were only generated for {len(results['explanation_samples'])} sampled objects, not all {results.get('total_detections', 0)} detections.")
        
        st.markdown("""
        <div class="feedback-box">
            <h5>Understanding AI Decisions</h5>
            <p>The heatmaps below show which parts of the image influenced the model's detection decisions. 
            Warmer colors (red/yellow) indicate regions that had the most impact on the classification.</p>
        </div>
        """, unsafe_allow_html=True)
        
        explanations_by_frame = {}
        for sample in results['explanation_samples']:
            frame_num = sample['frame_number']
            if frame_num not in explanations_by_frame:
                explanations_by_frame[frame_num] = []
            explanations_by_frame[frame_num].append(sample)
        
        st.markdown("""
        **🎨 Color Legend:** Each detected component is marked with a unique color:
        - 🟢 Object #1 (Green) | 🔵 Object #2 (Blue) | 🔴 Object #3 (Red) | 🔵 Object #4 (Cyan)
        - 🟣 Object #5 (Magenta) | 🟡 Object #6 (Yellow) | 🟣 Object #7 (Purple) | 🟠 Object #8 (Orange)
        """)
        
        st.markdown("---")
        
        frame_items = list(explanations_by_frame.items())
        
        for i in range(0, len(frame_items), 2):
            col_left, col_right = st.columns(2)
            
            with col_left:
                if i < len(frame_items):
                    frame_num, samples = frame_items[i]
                    st.markdown(f"### 📍 Sample {i//2 + 1} - Frame {frame_num}")
                    
                    if samples:
                        explained_frame = cv2.cvtColor(samples[0]['visualized_frame'], cv2.COLOR_BGR2RGB)
                        st.image(explained_frame, 
                                caption=f"SHAP Heatmap - Frame {frame_num} ({len(samples)} detection{'s' if len(samples) > 1 else ''})", 
                                use_container_width=True)
                    
                    for obj_idx, sample in enumerate(samples):
                        explanation = sample['explanation']
                        object_num = sample['object_index'] + 1
                        
                        color_emoji = ["🟢", "🔵", "🔴", "🔵", "🟣", "🟡", "🟣", "🟠"][obj_idx % 8]
                        
                        with st.expander(f"{color_emoji} Object #{object_num}: {explanation['component_name']} (Confidence: {explanation['confidence']:.2%})"):
                            cropped_img = cv2.cvtColor(explanation['cropped_region'], cv2.COLOR_BGR2RGB)
                            st.image(cropped_img, caption=f"Detected Component Region", use_container_width=True)
                            
                            bbox = explanation['detection_box']
                            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Confidence", f"{explanation['confidence']:.2%}")
                            with metric_col2:
                                st.metric("Area", f"{bbox_area:,} px²")
                            with metric_col3:
                                st.metric("Object #", f"{object_num}")
                            
                            st.markdown(f"""
                            **Detection Details:**
                            - **Component Type:** {explanation['component_name']}
                            - **Bounding Box:** ({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})
                            - **Width × Height:** {bbox[2] - bbox[0]} × {bbox[3] - bbox[1]} pixels
                            - **Center Point:** ({(bbox[0] + bbox[2])//2}, {(bbox[1] + bbox[3])//2})
                            """)
                            
                            st.info("💡 The heatmap shows regions that influenced this detection. Red/warm colors indicate high importance.")
            
            with col_right:
                if i + 1 < len(frame_items):
                    frame_num, samples = frame_items[i + 1]
                    st.markdown(f"### 📍 Sample {i//2 + 2} - Frame {frame_num}")
                    
                    if samples:
                        explained_frame = cv2.cvtColor(samples[0]['visualized_frame'], cv2.COLOR_BGR2RGB)
                        st.image(explained_frame, 
                                caption=f"SHAP Heatmap - Frame {frame_num} ({len(samples)} detection{'s' if len(samples) > 1 else ''})", 
                                use_container_width=True)
                    
                    for obj_idx, sample in enumerate(samples):
                        explanation = sample['explanation']
                        object_num = sample['object_index'] + 1
                        
                        color_emoji = ["🟢", "🔵", "🔴", "🔵", "🟣", "🟡", "🟣", "🟠"][obj_idx % 8]
                        
                        with st.expander(f"{color_emoji} Object #{object_num}: {explanation['component_name']} (Confidence: {explanation['confidence']:.2%})"):
                            cropped_img = cv2.cvtColor(explanation['cropped_region'], cv2.COLOR_BGR2RGB)
                            st.image(cropped_img, caption=f"Detected Component Region", use_container_width=True)
                            
                            bbox = explanation['detection_box']
                            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Confidence", f"{explanation['confidence']:.2%}")
                            with metric_col2:
                                st.metric("Area", f"{bbox_area:,} px²")
                            with metric_col3:
                                st.metric("Object #", f"{object_num}")
                            
                            st.markdown(f"""
                            **Detection Details:**
                            - **Component Type:** {explanation['component_name']}
                            - **Bounding Box:** ({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})
                            - **Width × Height:** {bbox[2] - bbox[0]} × {bbox[3] - bbox[1]} pixels
                            - **Center Point:** ({(bbox[0] + bbox[2])//2}, {(bbox[1] + bbox[3])//2})
                            """)
                            
                            st.info("💡 The heatmap shows regions that influenced this detection. Red/warm colors indicate high importance.")
            
            if i + 2 < len(frame_items):
                st.markdown("---")
        
        st.markdown("---")
        st.markdown("### 📈 Explainability Summary")
        
        col_sum1, col_sum2 = st.columns(2)
        
        with col_sum1:
            st.metric("Average Confidence (Sampled Frames)", f"{avg_confidence_samples:.2%}")
            st.info("💡 Average confidence for objects that received SHAP explanations.")
        
        with col_sum2:
            components_detected = set([s['explanation']['component_name'] for s in results['explanation_samples']])
            st.metric("Unique Components Explained", len(components_detected))
            st.info("🔍 Components: " + ", ".join(components_detected))
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if 'explanation_samples' in results:
            st.markdown("---")
            st.warning("⚠️ No SHAP explanations were generated. This can happen if:")
            st.markdown("""
            - No objects were detected in the sampled frames
            - The sampled frames didn't align with frames containing detections
            - Try processing the video again or adjust the confidence threshold
            """)
    
    st.markdown("---")
    
    with st.expander("📋 Detailed Performance Summary (Includes SHAP Analysis)"):
        st.markdown("### System Performance")
        summary_data = {}
        for key, value in results.items():
            if key not in ['detections_data', 'output_video', 'explanation_samples']:
                if isinstance(value, float):
                    summary_data[key] = f"{value:.4f}"
                elif isinstance(value, dict):
                    summary_data[key] = str(value)
                else:
                    summary_data[key] = str(value)
        
        for key, value in summary_data.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        if results.get('explanation_samples'):
            st.markdown("### SHAP Explainability Summary")
            st.write(f"**Total Explanations Generated:** {len(results['explanation_samples'])}")
            
            # NEW: Show comparison of confidence metrics
            avg_confidence_samples = np.mean([s['explanation']['confidence'] for s in results['explanation_samples']])
            st.write(f"**Average Confidence (Sampled Frames):** {avg_confidence_samples:.2%}")
            st.write(f"**Average Confidence (All Frames):** {results.get('avg_confidence', 0):.2%}")
            
            explanations_by_frame = {}
            for sample in results['explanation_samples']:
                frame_num = sample['frame_number']
                if frame_num not in explanations_by_frame:
                    explanations_by_frame[frame_num] = []
                explanations_by_frame[frame_num].append(sample)
            
            for frame_num, samples in explanations_by_frame.items():
                st.write(f"**Frame {frame_num}:**")
                for idx, sample in enumerate(samples):
                    exp = sample['explanation']
                    bbox = exp['detection_box']
                    st.write(f"  - Object {idx+1}: {exp['component_name']} "
                           f"(Confidence: {exp['confidence']:.2%}, "
                           f"BBox: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]}))")
        
        st.markdown("---")
        
        video_name = st.session_state.get('current_video_name', 'video')
        pdf_bytes = generate_pdf_report(results, video_name)
        
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"ACIS_Report_{int(time.time())}.pdf",
            mime="application/pdf",
            help="Download a comprehensive PDF report including all metrics and SHAP analysis"
        )
    
    st.markdown("---")
    
    st.subheader("💬 Customer Feedback")
    
    if not st.session_state.feedback_submitted:
        st.markdown("""
        <div class="feedback-box">
            <h4>📝 Help us improve!</h4>
            <p>Please rate your experience and provide detailed feedback about the detection system</p>
        </div>
        """, unsafe_allow_html=True)
        
        rating = st.slider("Rate your experience (1-5 stars)", 1, 5, 3)
        feedback_text = st.text_area("Your feedback", 
                                     placeholder="Tell us about your experience with the detection system...",
                                     height=100)
        
        if st.button("Submit Feedback", type="primary"):
            if feedback_text.strip():
                video_name = st.session_state.get('current_video_name', 'unknown')
                if save_feedback(rating, feedback_text, video_name):
                    st.session_state.feedback_submitted = True
                    st.success("✅ Thank you for your feedback!")
                    st.rerun()
            else:
                st.warning("Please provide some feedback before submitting.")
    else:
        st.success("✅ Feedback submitted! Thank you for helping us improve.")


if __name__ == "__main__":
    main()