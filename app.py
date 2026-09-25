"""
Interactive Web Application for Real-Time Pothole Detection & Hazard Assessment
Built with Streamlit and Ultralytics YOLO.
"""

import os
import io
import time
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Pothole Detection & Road Hazard System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF5722;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #78909C;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #FF5722;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛣️ YOLO Pothole Detection & Road Safety System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated deep learning road inspection for municipal management, dashcam warnings, and smart city infrastructure.</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Model Configuration")

weights_opt = ["weights/best.pt", "yolo11s.pt", "yolo11n.pt", "yolov8s.pt"]
selected_weights = st.sidebar.selectbox("Select Model Weights", weights_opt, index=0)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU NMS Threshold", 0.1, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Hazard Severity Legend")
st.sidebar.markdown("🔴 **CRITICAL**: Area > 8% (High vehicle damage risk)")
st.sidebar.markdown("🟠 **MODERATE**: Area 2.5% - 8% (Requires maintenance)")
st.sidebar.markdown("🟡 **MINOR**: Area < 2.5% (Early surface defect)")

@st.cache_resource
def load_yolo_model(weights_path):
    if not os.path.exists(weights_path) and weights_path.endswith("best.pt"):
        st.warning(f"Custom checkpoint '{weights_path}' not found yet. Loading pre-trained base model.")
        return YOLO("yolo11s.pt")
    return YOLO(weights_path)

try:
    model = load_yolo_model(selected_weights)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = YOLO("yolo11s.pt")

tab1, tab2, tab3 = st.tabs(["📷 Image Inspection", "🎥 Video / Dashcam Analysis", "ℹ️ System Info & Metrics"])

with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("1. Provide Image")
        source_mode = st.radio("Choose Input Mode:", ["Upload Image", "Pick Sample from Dataset"], horizontal=True)
        
        input_image = None
        if source_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload road photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert("RGB")
        else:
            sample_dir = Path("dataset/images/test")
            if sample_dir.exists():
                sample_files = list(sample_dir.glob("*.jpg"))[:15]
                if sample_files:
                    sample_choice = st.selectbox("Select Test Image", [f.name for f in sample_files])
                    input_image = Image.open(sample_dir / sample_choice).convert("RGB")
                else:
                    st.info("No sample images found in dataset/images/test")
            else:
                st.info("Dataset directory not found. Please upload an image.")
                
        if input_image:
            st.image(input_image, caption="Original Input Road Surface", use_container_width=True)

    with col_right:
        st.subheader("2. AI Detection & Diagnostics")
        if input_image is not None:
            img_np = np.array(input_image)
            orig_h, orig_w = img_np.shape[:2]
            frame_area = orig_h * orig_w
            
            with st.spinner("Analyzing road surface..."):
                t0 = time.time()
                results = model.predict(img_np, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
                latency_ms = (time.time() - t0) * 1000
                
            annotated_frame = img_np.copy()
            detections = []
            
            for box in results.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                bw = bx2 - bx1
                bh = by2 - by1
                box_area = bw * bh
                ratio = (box_area / frame_area) * 100.0
                
                if ratio >= 8.0:
                    sev, color = "CRITICAL", (255, 0, 0)
                elif ratio >= 2.5:
                    sev, color = "MODERATE", (255, 140, 0)
                else:
                    sev, color = "MINOR", (255, 215, 0)
                    
                detections.append({
                    "Pothole ID": len(detections) + 1,
                    "Confidence": f"{confidence * 100:.1f}%",
                    "Severity": sev,
                    "Road Area Coverage": f"{ratio:.2f}%",
                    "BBox [x1, y1, x2, y2]": f"[{bx1}, {by1}, {bx2}, {by2}]"
                })
                
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 3)
                label = f"Pothole {confidence:.2f} [{sev}]"
                cv2.putText(annotated_frame, label, (bx1, max(25, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                
            st.image(annotated_frame, caption="YOLO Annotated Detections", use_container_width=True)
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Potholes Detected", len(detections))
            crit_count = sum(1 for d in detections if d["Severity"] == "CRITICAL")
            m2.metric("Critical Hazards", crit_count)
            m3.metric("Inference Latency", f"{latency_ms:.1f} ms")
            
            if detections:
                st.markdown("### 📋 Diagnostic Telemetry")
                df_det = pd.DataFrame(detections)
                st.dataframe(df_det, use_container_width=True)
                
                # Download button
                csv_buffer = io.StringIO()
                df_det.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Telemetry CSV",
                    data=csv_buffer.getvalue(),
                    file_name="pothole_telemetry.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ No potholes detected above threshold. Road surface appears clear.")

with tab2:
    st.subheader("Dashcam / Video Hazard Detection")
    st.markdown("Upload road inspection dashcam video footage to evaluate real-time hazard tracking.")
    vid_file = st.file_uploader("Upload video (MP4, AVI)", type=["mp4", "avi", "mov"])
    if vid_file:
        st.info("Video inference engine is ready. For high-speed hardware acceleration on local video files, run: `python detect.py --source your_video.mp4`.")

with tab3:
    st.subheader("System Architecture & Model Specifications")
    st.markdown("""
    - **Detection Engine:** Ultralytics YOLO11 / YOLOv8
    - **Resolution:** 640x640 multi-scale pyramid
    - **Backbone:** Modified CSPDarknet with C3k2 blocks & SPPF
    - **Attention:** C2PSA (Cross-Stage Partial with Spatial Attention)
    - **Loss Function:** Dynamic Task-Aligned Loss (CIoU + DFL + BCE)
    - **Target Class:** `pothole` (Class 0)
    """)
