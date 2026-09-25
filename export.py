"""
Model Export Script for Edge & Production Deployment
Exports trained YOLO model to ONNX, TensorRT, TorchScript, OpenVINO, CoreML, and TFLite.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def export_model(weights="weights/best.pt", format_type="onnx", imgsz=640, half=False, dynamic=False):
    print("=" * 65)
    print("  📦 Exporting YOLO Pothole Detection Model")
    print("=" * 65)
    
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"⚠️ Checkpoint '{weights}' not found. Using pretrained 'yolo11s.pt'")
        weights_path = "yolo11s.pt"
        
    model = YOLO(str(weights_path))
    
    print(f"Loading weights: {weights_path}")
    print(f"Target format:   {format_type.upper()}")
    print(f"Image size:      {imgsz}")
    print(f"FP16 half:       {half}")
    print(f"Dynamic shapes:  {dynamic}")
    
    export_path = model.export(
        format=format_type,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic
    )
    
    print("\n" + "=" * 65)
    print(f"  ✅ Model successfully exported to: {export_path}")
    print("=" * 65)
    return export_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export YOLO Pothole Detection Model for Deployment")
    parser.add_argument('--weights', type=str, default="weights/best.pt", help="Path to trained .pt weights")
    parser.add_argument('--format', type=str, default="onnx", 
                        choices=['onnx', 'engine', 'torchscript', 'openvino', 'coreml', 'tflite'],
                        help="Export format (onnx, engine, torchscript, openvino, coreml, tflite)")
    parser.add_argument('--imgsz', type=int, default=640, help="Input image size")
    parser.add_argument('--half', action='store_true', help="Use FP16 half precision")
    parser.add_argument('--dynamic', action='store_true', help="Support dynamic input dimensions")
    args = parser.parse_args()

    export_model(
        weights=args.weights,
        format_type=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic
    )
