"""
Visualize Ground Truth YOLO Annotations
Draws bounding boxes from labels onto images to verify annotation accuracy.
"""

import os
import random
import argparse
from pathlib import Path
import cv2

def draw_yolo_boxes(img, label_path, class_names):
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img, 0
        
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        cls_id = int(parts[0])
        x_c, y_c, bw, bh = map(float, parts[1:])
        
        # De-normalize coordinates
        bx1 = int((x_c - bw / 2.0) * w)
        by1 = int((y_c - bh / 2.0) * h)
        bx2 = int((x_c + bw / 2.0) * w)
        by2 = int((y_c + bh / 2.0) * h)
        
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        
        # Draw bounding box (vibrant red-orange for pothole hazard)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 69, 255), 2)
        
        # Label badge background
        badge_text = f"{cls_name}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, max(th + 6, by1)), (0, 69, 255), -1)
        cv2.putText(img, badge_text, (bx1 + 3, max(th + 2, by1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        count += 1
        
    return img, count

def visualize(dataset_dir="dataset", split="train", num_samples=6, output_dir="runs/visualize"):
    base = Path(dataset_dir)
    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"No images found in {img_dir}")
        return
        
    # Prefer images that actually have labels with >0 boxes
    valid_samples = []
    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            valid_samples.append(img_path)
            if len(valid_samples) >= num_samples * 3:
                break
                
    chosen = random.sample(valid_samples if valid_samples else images, min(num_samples, len(images)))
    class_names = ['pothole']
    
    print(f"Visualizing {len(chosen)} samples from '{split}' split...")
    for idx, img_path in enumerate(chosen):
        img = cv2.imread(str(img_path))
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        annotated_img, count = draw_yolo_boxes(img, str(lbl_path), class_names)
        
        save_path = out_dir / f"sample_{idx+1}_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), annotated_img)
        print(f"  Saved: {save_path.name} ({count} potholes)")
        
    print(f"\nAll verification images saved to: {out_dir.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Ground Truth YOLO Annotations")
    parser.add_argument('--dataset', type=str, default="dataset", help="Path to YOLO dataset root")
    parser.add_argument('--split', type=str, default="train", choices=['train', 'val', 'test'])
    parser.add_argument('--num', type=int, default=6, help="Number of random samples to visualize")
    parser.add_argument('--out', type=str, default="runs/visualize", help="Output directory")
    args = parser.parse_args()
    
    visualize(args.dataset, args.split, args.num, args.out)
