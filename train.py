import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

    data_path = Path(data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found at: {data_path}")
    print(f"📁 Dataset Config: {data_path}")
    print(f"🧠 Model Architecture: {model_name}")
    print(f"⚙️ Hyperparameters: Epochs={epochs}, ImgSz={imgsz}, Batch={batch}, Patience={patience}")

    model = YOLO(model_name)

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
        cos_lr=True,
        patience=patience,
        save=True,
        save_period=5,
        plots=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        close_mosaic=min(5, epochs // 4) if epochs > 5 else 0,
        workers=0 if sys.platform == 'win32' else 4,
        verbose=True
    )

    print("\n" + "=" * 65)
    print("  ✅ Training Completed Successfully!")
    print("=" * 65)

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
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
