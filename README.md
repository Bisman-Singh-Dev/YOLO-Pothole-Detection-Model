<div align="center">

# 🛣️ YOLO Pothole Detection & Road Hazard Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8%20%7C%2011-00FFFF.svg?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20Accelerated-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Bisman-Singh-Dev/YOLO-Pothole-Detection-Model?style=social)](https://github.com/Bisman-Singh-Dev/YOLO-Pothole-Detection-Model)

**An end-to-end, high-accuracy deep learning pipeline for real-time pothole detection, road surface hazard scoring, driver collision warning HUDs, and municipal road infrastructure maintenance.**

[Key Features](#-key-features) •
[Architecture](#-system-architecture) •
[Dataset](#-dataset-overview) •
[Quickstart](#-quick-start) •
[Training](#-training-the-model) •
[Inference](#-inference--real-time-detection) •
[Web App](#-interactive-web-dashboard) •
[Export](#-edge--hardware-deployment)

---

</div>

## 📌 Overview

Road anomalies and potholes cause billions of dollars in vehicular damage and thousands of accidents globally each year. This repository delivers an industrial-grade road damage detection solution powered by state-of-the-art **YOLO (YOLO11 / YOLOv8)** architectures. 

The pipeline is trained and validated on a curated dataset of **4,054 road images** with **5,601 ground-truth pothole annotations**, optimized specifically to distinguish complex asphalt cracks, irregular crater boundaries, shadows, wet roads, and varying weather conditions.

---

## ✨ Key Features

- 🎯 **State-of-the-Art Accuracy**: Employs YOLO11 with C3k2 building blocks, SPPF, and C2PSA (Cross-Stage Partial with Spatial Attention) for pinpoint pothole localization.
- 🛑 **Automated Hazard Severity Classifier**: Computes the relative road surface area occupied by each pothole and categorizes it into:
  - 🔴 **CRITICAL HAZARD** (Road area $\ge$ 8.0%): High blowout/suspension damage threat. Triggers immediate alert.
  - 🟠 **MODERATE RISK** (Road area 2.5% – 8.0%): Road degradation requiring maintenance.
  - 🟡 **MINOR DEFECT** (Road area < 2.5%): Early surface fissure.
- 🚗 **Real-Time Driver Assist HUD**: Visual head-up display overlay with live FPS counter, pothole count, and warning banners for dashcam integration.
- 📊 **Telemetry Logging**: Automatically logs detected potholes with timestamps, frame numbers, confidence scores, and bounding boxes into structured `CSV` and `JSON` files for municipal GIS mapping.
- 🌐 **Interactive Streamlit Web Dashboard**: Built-in drag-and-drop web application (`app.py`) for instantaneous inspection and reporting.
- ⚡ **Multi-Format Edge Export**: Easily convert models to **ONNX**, **TensorRT**, **TorchScript**, **OpenVINO**, **CoreML**, and **TFLite** for embedded vehicle deployment (Jetson Orin, Raspberry Pi, Android, iOS).

---

## 🏛️ System Architecture

```mermaid
flowchart LR
    A[Road Camera / Dashcam] --> B[Frame Capture & Normalization]
    B --> C[YOLO11 / YOLOv8 Backbone & C2PSA Attention]
    C --> D[Multi-Scale Feature Pyramid PAN-FPN]
    D --> E[Anchor-Free Detection Head]
    E --> F[Non-Maximum Suppression (NMS)]
    F --> G[Pothole BBoxes & Confidences]
    G --> H[Severity Engine (Critical / Moderate / Minor)]
    H --> I[Real-Time Driver HUD Alert]
    H --> J[Municipal GIS Telemetry Log]
```

---

## 📂 Dataset Overview

The dataset consists of **4,054 high-resolution road images** partitioned into standard train, validation, and test splits with **5,601 verified annotations**:

| Split | Images | Format | Description |
| :--- | :---: | :---: | :--- |
| **Train** | **2,749** | Normalized YOLO | Used for model optimization with mosaic & mixup augmentations |
| **Validation** | **654** | Normalized YOLO | Used for early stopping, learning rate scheduling & checkpointing |
| **Test** | **651** | Normalized YOLO | Unseen benchmark evaluation set |
| **Total** | **4,054** | **1 Class (`pothole`)** | Complete self-contained dataset included in `dataset/` |

### Annotation Format (`dataset/labels/`):
Each label `.txt` file follows standard YOLO object detection coordinates:
```text
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized between `0.0` and `1.0`.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Bisman-Singh-Dev/YOLO-Pothole-Detection-Model.git
cd YOLO-Pothole-Detection-Model
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Train the model with road-optimized hyperparameters (HSV jitter for asphalt lighting, perspective distortion, mosaic, and mixup):

```bash
# Recommended: Train with YOLO11 on GPU
python train.py --model yolo11s.pt --epochs 50 --batch 16 --device 0

# Lightweight version for ultra-fast edge inference:
python train.py --model yolo11n.pt --epochs 50 --batch 32 --device 0
```

### Training CLI Arguments:
| Argument | Default | Description |
| :--- | :---: | :--- |
| `--data` | `data.yaml` | Path to dataset configuration file |
| `--model` | `yolo11s.pt` | Model backbone (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolov8s.pt`) |
| `--epochs` | `50` | Total training iterations |
| `--batch` | `16` | Batch size |
| `--imgsz` | `640` | Input image resolution |
| `--device` | `0` | GPU device ID (`0`, `0,1`) or `cpu` |
| `--patience` | `15` | Early stopping epochs |
| `--optimizer` | `auto` | Optimizer (`SGD`, `Adam`, `AdamW`) |

All training checkpoints, loss curves, confusion matrices, and precision-recall graphs are automatically logged to `runs/detect/`.

---

## 🔍 Inference & Real-Time Detection

The inference engine in `detect.py` supports images, image directories, dashcam video files, and live camera feeds:

### 1. Test on Images
```bash
# Run detection on test split images
python detect.py --source dataset/images/test --conf 0.25 --save
```

### 2. Run on Dashcam Video
```bash
# Detect potholes in a driving video and render Driver HUD
python detect.py --source dashcam_video.mp4 --conf 0.30 --save
```

### 3. Run Live from Webcam or RTSP Stream
```bash
# Live webcam feed (0) with real-time preview HUD
python detect.py --source 0 --show
```

### Telemetry Output:
All detections generate structured records in `runs/detect/predict/pothole_telemetry.csv`:
```csv
frame,box,confidence,severity,coverage_pct
1,"[142, 380, 290, 460]",0.8924,CRITICAL,9.45
1,"[410, 320, 480, 370]",0.7412,MINOR,1.82
```

---

## 📊 Benchmarking & Evaluation

The model was evaluated against the unseen test split (**651 road images**, **833 potholes**) yielding strong localization accuracy and real-time inference speeds:

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Precision (P)** | **76.1%** (`0.7609`) | High detection confidence, minimal false alarms |
| **Recall (R)** | **63.8%** (`0.6380`) | Captures faint cracks and deep asphalt craters |
| **mAP @ 0.50** | **71.3%** (`0.7131`) | Mean Average Precision at IoU 0.50 |
| **mAP @ 0.50:0.95** | **36.6%** (`0.3657`) | Strict multi-threshold spatial localization |
| **Inference Latency** | **13.0 ms** (~76 FPS) | NVIDIA RTX 4050 GPU (Laptop) |
| **Weights Size** | **5.46 MB** | Compact, ideal for mobile/dashcam edge chips |

To re-run evaluation benchmarks:

```bash
python evaluate.py --weights weights/best.pt --split test
```

This generates:
- **mAP@0.50** and **mAP@0.50:0.95** scores
- **Precision-Recall (PR) Curves**
- **F1-Score Curves**
- **Normalized Confusion Matrix**

---

## 🖥️ Interactive Web Dashboard

Launch the Streamlit interactive dashboard to test individual images or videos with real-time sliders:

```bash
streamlit run app.py
```

Features:
- Live confidence & IoU threshold adjustment sliders
- Severity hazard color coding & distribution metrics
- One-click CSV telemetry report download

---

## 📦 Edge & Hardware Deployment

Export the trained model to standard production runtime engines:

```bash
# ONNX for general cross-platform deployment
python export.py --weights weights/best.pt --format onnx

# TensorRT for NVIDIA Jetson / Orin automotive edge computers
python export.py --weights weights/best.pt --format engine --half

# TorchScript for C++ LibTorch integration
python export.py --weights weights/best.pt --format torchscript

# OpenVINO for Intel CPU / iGPU edge devices
python export.py --weights weights/best.pt --format openvino

# TFLite for Android / mobile edge inspection
python export.py --weights weights/best.pt --format tflite
```

---

## 🗂️ Project Directory Tree

```text
YOLO-Pothole-Detection-Model/
├── dataset/
│   ├── images/
│   │   ├── train/            # 2,749 training images
│   │   ├── val/              # 654 validation images
│   │   └── test/             # 651 test images
│   ├── labels/
│   │   ├── train/            # 2,749 YOLO annotation files
│   │   ├── val/              # 654 YOLO annotation files
│   │   └── test/             # 651 YOLO annotation files
│   └── data.yaml             # Dataset specification
├── scripts/
│   ├── prepare_dataset.py    # Raw dataset converter & preprocessor
│   └── visualize_annotations.py # Verification visualizer for ground-truth labels
├── weights/                  # Model weights (best.pt)
├── train.py                  # High-accuracy YOLO training script
├── evaluate.py               # Benchmark & validation script
├── detect.py                 # Real-time inference & Driver HUD engine
├── export.py                 # Multi-format deployment exporter
├── app.py                    # Interactive Streamlit web app
├── data.yaml                 # Root YOLO data configuration
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusions
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## 🛡️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Developed by **[Bisman-Singh-Dev](https://github.com/Bisman-Singh-Dev)**. Contributions and feature suggestions are welcome via pull requests and issues!
