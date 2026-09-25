"""
Live Webcam Testing & Interactive Hazard Demo for YOLO Pothole Detection Model.
Run simply with:
    python test_camera.py

Controls:
    [Q] or [ESC] : Exit the camera feed
    [S]          : Save high-resolution annotated snapshot
    [+] or [=]   : Increase confidence threshold (+0.05)
    [-] or [_]   : Decrease confidence threshold (-0.05)
    [H]          : Toggle HUD overlay
"""

import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def classify_hazard(box_area, frame_area):
    """Categorize pothole into severity levels based on road surface ratio."""
    ratio = (box_area / frame_area) * 100.0
    if ratio >= 8.0:
        return "CRITICAL", (0, 0, 255), ratio       # BGR: Red
    elif ratio >= 2.5:
        return "MODERATE", (0, 140, 255), ratio     # BGR: Orange
    else:
        return "MINOR", (0, 215, 255), ratio        # BGR: Yellow/Amber

def draw_driver_hud(frame, detections, fps, conf_thresh, show_hud=True):
    """Draw an advanced automotive-style HUD overlay on camera frame."""
    if not show_hud:
        return frame
        
    h, w = frame.shape[:2]
    
    # Top Telemetry Header Bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # HUD Text
    cv2.putText(frame, "AI ROAD HAZARD DETECTION [LIVE CAMERA]", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)
    
    pothole_count = len(detections)
    crit_count = sum(1 for d in detections if d['severity'] == "CRITICAL")
    
    stat_msg = f"Detected: {pothole_count} | Critical: {crit_count} | Conf: {conf_thresh:.2f}"
    cv2.putText(frame, stat_msg, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    
    # Critical Alert Banner (if critical pothole is detected)
    if crit_count > 0:
        banner_w = min(480, w - 40)
        bx1 = (w - banner_w) // 2
        bx2 = bx1 + banner_w
        cv2.rectangle(frame, (bx1, h - 85), (bx2, h - 45), (0, 0, 220), -1)
        cv2.rectangle(frame, (bx1, h - 85), (bx2, h - 45), (255, 255, 255), 2)
        cv2.putText(frame, "CRITICAL: POTHOLE HAZARD AHEAD!", (bx1 + 20, h - 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
    # Bottom Control Hints
    controls_text = "[Q] Quit | [S] Snapshot | [+/-] Adjust Conf | [H] Toggle HUD"
    cv2.putText(frame, controls_text, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
                
    return frame

def open_webcam(cam_id=0):
    """Open camera with Windows DirectShow optimization."""
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_id)
    else:
        cap = cv2.VideoCapture(cam_id)
    return cap

def run_camera_test(weights="weights/best.pt", cam_id=0, conf=0.25):
    print("=" * 65)
    print("  📹 Launching YOLO Pothole Detection Live Camera Test")
    print("=" * 65)
    
    # Locate model checkpoint
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"⚠️ Warning: Checkpoint '{weights}' not found. Loading base model 'yolo11s.pt'.")
        weights_path = Path("yolo11s.pt")
        
    print(f"Loading weights: {weights_path}")
    model = YOLO(str(weights_path))
    
    # Output directory for snapshots
    snap_dir = Path("runs/detect/camera_snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to camera index {cam_id}...")
    cap = open_webcam(cam_id)
    
    if not cap.isOpened():
        # Try camera index 1 if 0 failed
        print(f"⚠️ Could not open camera {cam_id}. Testing camera index 1...")
        cap = open_webcam(1)
        if cap.isOpened():
            cam_id = 1
            print("Connected to camera index 1.")
            
    if not cap.isOpened():
        print("\n❌ Error: No webcam detected or camera access is blocked.")
        print("Possible causes:")
        print("  1. No webcam is connected to this device.")
        print("  2. Another program (Zoom, MS Teams, Chrome) is currently using the camera.")
        print("  3. Windows Camera Privacy Settings blocked access to desktop apps.")
        print("\n💡 You can test the model on test dataset images or video instead:")
        print("   python detect.py --source dataset/images/test/pothole_dataset_1_train_image10.jpg")
        print("   python detect.py --source dashcam_video.mp4")
        return
        
    # Set camera capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    frame_area = actual_w * actual_h
    
    print(f"✅ Camera online! Resolution: {actual_w}x{actual_h}")
    print("\nInteractive Controls in the camera window:")
    print("  [Q] or [ESC] - Quit")
    print("  [S]         - Save snapshot to runs/detect/camera_snapshots/")
    print("  [+] or [=]   - Increase confidence threshold")
    print("  [-] or [_]   - Decrease confidence threshold")
    print("  [H]         - Toggle HUD overlay on/off")
    print("=" * 65)
    
    prev_time = time.time()
    show_hud = True
    current_conf = conf
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Camera frame capture failed. Exiting.")
                break
                
            curr_time = time.time()
            fps = 1.0 / max(1e-5, (curr_time - prev_time))
            prev_time = curr_time
            
            # Real-time YOLO inference
            results = model.predict(frame, conf=current_conf, iou=0.45, imgsz=640, verbose=False)[0]
            
            frame_dets = []
            for box in results.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                bw = bx2 - bx1
                bh = by2 - by1
                box_area = bw * bh
                
                sev, color, ratio = classify_hazard(box_area, frame_area)
                frame_dets.append({
                    'box': [bx1, by1, bx2, by2],
                    'confidence': confidence,
                    'severity': sev,
                    'ratio': ratio
                })
                
                # Draw styled bounding box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                label = f"Pothole {confidence:.2f} [{sev}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, max(th + 6, by1)), color, -1)
                cv2.putText(frame, label, (bx1 + 3, max(th + 2, by1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            
            # Render HUD
            frame = draw_driver_hud(frame, frame_dets, fps, current_conf, show_hud)
            
            cv2.imshow("YOLO Pothole Detector - Live Camera Feed", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key in [ord('q'), 27]: # 'q' or ESC
                print("\nCamera test closed by user.")
                break
            elif key in [ord('s'), ord('S')]:
                snap_path = snap_dir / f"pothole_snap_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), frame)
                print(f"📸 Snapshot saved: {snap_path}")
            elif key in [ord('+'), ord('=')]:
                current_conf = min(0.95, current_conf + 0.05)
                print(f"Confidence threshold set to: {current_conf:.2f}")
            elif key in [ord('-'), ord('_')]:
                current_conf = max(0.05, current_conf - 0.05)
                print(f"Confidence threshold set to: {current_conf:.2f}")
            elif key in [ord('h'), ord('H')]:
                show_hud = not show_hud
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera resource released cleanly.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Live Webcam Pothole Detection Tester")
    parser.add_argument('--weights', type=str, default="weights/best.pt", help="Path to model weights")
    parser.add_argument('--cam-id', type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()
    
    run_camera_test(weights=args.weights, cam_id=args.cam_id, conf=args.conf)
