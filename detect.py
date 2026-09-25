"""
Real-Time YOLO Pothole Detection & Road Hazard Analysis Engine
Supports single images, directories, video files (dashcams), and live webcam/RTSP feeds.
Includes severity classification, driver HUD alert overlay, and structured telemetry export.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def classify_severity(box_area, frame_area):
    """
    Classify pothole severity based on relative road coverage area.
    """
    ratio = (box_area / frame_area) * 100.0
    if ratio >= 8.0:
        return "CRITICAL", (0, 0, 255), ratio       # Bright Red
    elif ratio >= 2.5:
        return "MODERATE", (0, 140, 255), ratio     # Orange
    else:
        return "MINOR", (0, 215, 255), ratio        # Amber/Yellow

def draw_hud(frame, detections, fps=0.0):
    """
    Renders a driver-assist HUD on the frame with hazard telemetry.
    """
    h, w = frame.shape[:2]
    
    # Top telemetry panel bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Text headers
    cv2.putText(frame, "ROAD HAZARD DETECTION SYSTEM", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)
    
    pothole_count = len(detections)
    crit_count = sum(1 for d in detections if d['severity'] == "CRITICAL")
    
    status_text = f"Potholes Detected: {pothole_count} | Critical Hazards: {crit_count}"
    cv2.putText(frame, status_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    
    # Alert banner if critical pothole is detected
    if crit_count > 0:
        alert_w = min(500, w - 40)
        ax1 = (w - alert_w) // 2
        ax2 = ax1 + alert_w
        cv2.rectangle(frame, (ax1, h - 60), (ax2, h - 15), (0, 0, 200), -1)
        cv2.rectangle(frame, (ax1, h - 60), (ax2, h - 15), (255, 255, 255), 2)
        cv2.putText(frame, "WARNING: SEVERE ROAD DAMAGE AHEAD", (ax1 + 25, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        
    return frame

def run_detection(
    source="dataset/images/test",
    weights="weights/best.pt",
    conf=0.25,
    iou=0.45,
    imgsz=640,
    device=None,
    save=True,
    show=False,
    output_dir="runs/detect/predict",
    export_csv=True
):
    print("=" * 65)
    print("  🔍 Initializing YOLO Pothole Detection Inference")
    print("=" * 65)
    
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"⚠️ Checkpoint '{weights}' not found. Falling back to pretrained 'yolo11s.pt'")
        weights_path = "yolo11s.pt"
        
    model = YOLO(str(weights_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    is_cam = source.isdigit() or source.startswith('rtsp://') or source.startswith('http://')
    is_video = not is_cam and Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    
    telemetry_records = []
    
    if is_cam or is_video:
        cap_src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        writer = None
        if save:
            out_vid = out_dir / "detection_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_vid), fourcc, src_fps, (w, h))
            print(f"🎥 Saving annotated video to: {out_vid}")
            
        frame_idx = 0
        prev_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                curr_time = time.time()
                fps = 1.0 / max(1e-5, (curr_time - prev_time))
                prev_time = curr_time
                
                # Model inference
                results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
                
                frame_area = w * h
                frame_dets = []
                
                for box in results.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    box_w = bx2 - bx1
                    box_h = by2 - by1
                    box_area = box_w * box_h
                    
                    sev, color, ratio = classify_severity(box_area, frame_area)
                    frame_dets.append({
                        'frame': frame_idx,
                        'box': [bx1, by1, bx2, by2],
                        'confidence': round(confidence, 4),
                        'severity': sev,
                        'coverage_pct': round(ratio, 2)
                    })
                    
                    # Draw detection box with severity styling
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    label = f"Pothole {confidence:.2f} ({sev})"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, max(th + 6, by1)), color, -1)
                    cv2.putText(frame, label, (bx1 + 3, max(th + 2, by1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                telemetry_records.extend(frame_dets)
                frame = draw_hud(frame, frame_dets, fps)
                
                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow("YOLO Pothole Detection HUD", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
                
    else:
        # Directory or single image
        src_path = Path(source)
        if src_path.is_file():
            img_paths = [src_path]
        else:
            img_paths = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png")) + list(src_path.glob("*.jpeg"))
            
        print(f"Found {len(img_paths)} image(s) for processing...")
        for img_p in img_paths:
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            frame_area = w * h
            
            results = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
            frame_dets = []
            
            for box in results.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                box_area = (bx2 - bx1) * (by2 - by1)
                
                sev, color, ratio = classify_severity(box_area, frame_area)
                frame_dets.append({
                    'image': img_p.name,
                    'box': [bx1, by1, bx2, by2],
                    'confidence': round(confidence, 4),
                    'severity': sev,
                    'coverage_pct': round(ratio, 2)
                })
                
                cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
                label = f"Pothole {confidence:.2f} [{sev}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, max(th + 6, by1)), color, -1)
                cv2.putText(img, label, (bx1 + 3, max(th + 2, by1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
            img = draw_hud(img, frame_dets)
            telemetry_records.extend(frame_dets)
            
            if save:
                save_p = out_dir / f"det_{img_p.name}"
                cv2.imwrite(str(save_p), img)
                
        print(f"Processed {len(img_paths)} images. Annotated results saved to: {out_dir}")
        
    # Export structured telemetry logs
    if export_csv and telemetry_records:
        csv_file = out_dir / "pothole_telemetry.csv"
        json_file = out_dir / "pothole_telemetry.json"
        
        df_log = pd.DataFrame(telemetry_records)
        df_log.to_csv(csv_file, index=False)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(telemetry_records, f, indent=2)
            
        print(f"📊 Telemetry exported to:")
        print(f"   CSV:  {csv_file}")
        print(f"   JSON: {json_file}")
        
    return telemetry_records

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO Pothole Detection Inference")
    parser.add_argument('--source', type=str, default="dataset/images/test", help="Image path, directory, video path, or webcam ID (0)")
    parser.add_argument('--weights', type=str, default="weights/best.pt", help="Path to model weights")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold")
    parser.add_argument('--iou', type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument('--imgsz', type=int, default=640, help="Inference resolution")
    parser.add_argument('--device', type=str, default=None, help="Device (e.g. 0, cpu)")
    parser.add_argument('--show', action='store_true', help="Display live rendering window")
    parser.add_argument('--no-save', dest='save', action='store_false', help="Disable saving results")
    parser.add_argument('--output', type=str, default="runs/detect/predict", help="Output directory")
    args = parser.parse_args()

    run_detection(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        show=args.show,
        output_dir=args.output
    )
