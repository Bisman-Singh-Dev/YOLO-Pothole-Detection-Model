"""
High-Accuracy YOLO Pothole Detection Training Pipeline
Optimized for road asphalt anomalies, irregular crater contours, and varying lighting.
Supports YOLO11, YOLOv8, YOLOv9, and YOLOv10 architectures via Ultralytics.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def train(
    data="data.yaml",
    model_name="yolo11s.pt",
    epochs=50,
    imgsz=640,
    batch=16,
    device=None,
    project="runs/detect",
    name="pothole_yolo11",
    optimizer="auto",
    lr0=0.01,
    patience=15,
    save_dir="weights"
):
    print("=" * 65)
    print("  🚀 Starting YOLO Pothole Detection Training Pipeline")
    print("=" * 65)
    
    # Check compute device
    if device is None:
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🔥 Hardware Acceleration: NVIDIA GPU detected -> {gpu_name}")
            print(f"   VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            device = 'cpu'
            print("⚠️ Hardware Acceleration: CUDA not available. Running on CPU.")
    else:
        print(f"Hardware Device configured: {device}")

    # Resolve data path
    data_path = Path(data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found at: {data_path}")
    print(f"📁 Dataset Config: {data_path}")
    print(f"🧠 Model Architecture: {model_name}")
    print(f"⚙️ Hyperparameters: Epochs={epochs}, ImgSz={imgsz}, Batch={batch}, Patience={patience}")

    # Load YOLO model
    model = YOLO(model_name)

    # Train model with asphalt-optimized augmentations
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        optimizer=optimizer,
        lr0=lr0,
        lrf=0.01,
        patience=patience,
        save=True,
        save_period=5,
        plots=True,
        # Augmentations tuned specifically for road & lighting conditions:
        hsv_h=0.015,       # Slight hue jitter
        hsv_s=0.7,         # Wet/dry asphalt saturation variation
        hsv_v=0.4,         # Shadow and daylight brightness jitter
        degrees=5.0,       # Minor camera pitch/roll tilt
        translate=0.1,     # Panning jitter
        scale=0.5,         # Scale variations (near vs distant potholes)
        perspective=0.0005,# Road surface perspective distortion
        fliplr=0.5,        # Horizontal flip
        mosaic=1.0,        # Multi-scale contextual mosaic
        mixup=0.1,         # Image blending
        close_mosaic=10,   # Disable mosaic in final 10 epochs for fine-tuning
        workers=0 if sys.platform == 'win32' else 4,
        verbose=True
    )

    print("\n" + "=" * 65)
    print("  ✅ Training Completed Successfully!")
    print("=" * 65)

    # Copy best weights to convenient weights/ directory
    best_pt = Path(project) / name / "weights" / "best.pt"
    if best_pt.exists():
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_best = target_dir / "best.pt"
        import shutil
        shutil.copy2(str(best_pt), str(dest_best))
        print(f"🏆 Best model checkpoint saved to: {dest_best.resolve()}")
    else:
        print(f"Check training output in: {Path(project) / name}")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO Model for High-Accuracy Pothole Detection")
    parser.add_argument('--data', type=str, default="data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument('--model', type=str, default="yolo11s.pt", help="Pretrained model (e.g. yolo11n.pt, yolo11s.pt, yolo11m.pt, yolov8s.pt)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument('--imgsz', type=int, default=640, help="Image resolution for training (default: 640)")
    parser.add_argument('--batch', type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument('--device', type=str, default=None, help="Device to run on (e.g. 0, cpu, 0,1)")
    parser.add_argument('--optimizer', type=str, default="auto", choices=['auto', 'SGD', 'Adam', 'AdamW', 'RMSProp'], help="Optimizer")
    parser.add_argument('--lr0', type=float, default=0.01, help="Initial learning rate")
    parser.add_argument('--patience', type=int, default=15, help="Early stopping patience")
    parser.add_argument('--project', type=str, default="runs/detect", help="Project save directory")
    parser.add_argument('--name', type=str, default="pothole_yolo11", help="Experiment name")
    parser.add_argument('--save-dir', type=str, default="weights", help="Directory to save final best model")
    args = parser.parse_args()

    train(
        data=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        save_dir=args.save_dir
    )
