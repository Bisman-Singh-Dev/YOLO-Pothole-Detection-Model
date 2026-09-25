"""
Comprehensive Model Evaluation & Benchmarking Script
Calculates Precision, Recall, mAP@50, mAP@50-95, and inference latency on the test split.
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def evaluate(
    weights="weights/best.pt",
    data="data.yaml",
    split="test",
    imgsz=640,
    batch=16,
    device=None,
    project="runs/detect",
    name="val_test"
):
    print("=" * 65)
    print("  📊 Evaluating YOLO Pothole Detection Model")
    print("=" * 65)
    
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"⚠️ Warning: Checkpoint '{weights}' not found. Using pretrained 'yolo11s.pt' as baseline.")
        weights_path = "yolo11s.pt"
    else:
        print(f"📦 Model Weights: {weights_path.resolve()}")

    model = YOLO(str(weights_path))

    import sys
    metrics = model.val(
        data=data,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0 if sys.platform == 'win32' else 4,
        project=project,
        name=name,
        plots=True,
        save_json=True
    )

    print("\n" + "=" * 65)
    print("  🎯 Benchmark Results Summary")
    print("=" * 65)
    print(f"Split:               {split}")
    print(f"Precision (P):       {metrics.box.mp:.4f}")
    print(f"Recall (R):          {metrics.box.mr:.4f}")
    print(f"mAP @ 0.50:          {metrics.box.map50:.4f}")
    print(f"mAP @ 0.50-0.95:     {metrics.box.map:.4f}")
    print(f"Speed (Preprocess):  {metrics.speed.get('preprocess', 0):.2f} ms")
    print(f"Speed (Inference):   {metrics.speed.get('inference', 0):.2f} ms")
    print(f"Speed (Postprocess): {metrics.speed.get('postprocess', 0):.2f} ms")
    print("=" * 65)
    print(f"Artifacts and plots saved to: {Path(project) / name}")

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Trained YOLO Model on Pothole Dataset")
    parser.add_argument('--weights', type=str, default="weights/best.pt", help="Path to trained weights file")
    parser.add_argument('--data', type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument('--split', type=str, default="test", choices=['test', 'val', 'train'], help="Dataset split to evaluate")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--device', type=str, default=None, help="Device (e.g. 0, cpu)")
    parser.add_argument('--project', type=str, default="runs/detect", help="Project directory")
    parser.add_argument('--name', type=str, default="evaluation_test", help="Evaluation folder name")
    args = parser.parse_args()

    evaluate(
        weights=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )
