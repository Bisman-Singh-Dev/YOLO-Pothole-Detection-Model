"""
Prepare YOLO Pothole Detection Dataset
Converts raw images, binary segmentation masks, and splits.csv into standard YOLO format.
"""

import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

def process_sample(args):
    row, src_img_dir, src_mask_dir, out_img_dir, out_lbl_dir, target_size, min_area = args
    img_name = row['Image']
    mask_name = row['Mask']
    split = row['Split']
    
    stem = Path(img_name).stem
    src_img_path = os.path.join(src_img_dir, img_name)
    src_mask_path = os.path.join(src_mask_dir, mask_name)
    
    dst_img_path = os.path.join(out_img_dir, split, f"{stem}.jpg")
    dst_lbl_path = os.path.join(out_lbl_dir, split, f"{stem}.txt")
    
    img = cv2.imread(src_img_path)
    if img is None:
        return 0, 0
    
    mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0, 0
        
    orig_h, orig_w = mask.shape[:2]
    
    # Extract pothole contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_labels = []
    potholes_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Normalized coordinates relative to original mask/image
        x_center = (bx + bw / 2.0) / orig_w
        y_center = (by + bh / 2.0) / orig_h
        w_norm = bw / float(orig_w)
        h_norm = bh / float(orig_h)
        
        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))
        
        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        potholes_count += 1
        
    # Resize image if target_size is specified
    if target_size:
        if (orig_w, orig_h) != (target_size, target_size):
            img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img
        cv2.imwrite(dst_img_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    else:
        cv2.imwrite(dst_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
    # Save YOLO format annotations
    with open(dst_lbl_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_labels))
        
    return 1, potholes_count

def convert_dataset(src_dir, out_dir, target_size=640, min_area=15, num_workers=8):
    src_path = Path(src_dir)
    out_path = Path(out_dir)
    
    splits_csv = src_path / "splits.csv"
    src_images = src_path / "images"
    src_masks = src_path / "masks"
    
    if not splits_csv.exists():
        raise FileNotFoundError(f"splits.csv not found in {src_dir}")
        
    df = pd.read_csv(splits_csv)
    print(f"Loaded splits.csv with {len(df)} records.")
    print("Split distribution:\n", df['Split'].value_counts())
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    tasks = []
    for _, row in df.iterrows():
        tasks.append((
            row,
            str(src_images),
            str(src_masks),
            str(out_path / "images"),
            str(out_path / "labels"),
            target_size,
            min_area
        ))
        
    print(f"\nProcessing {len(tasks)} samples using {num_workers} worker threads...")
    total_imgs = 0
    total_potholes = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for imgs_done, potholes_done in tqdm(executor.map(process_sample, tasks), total=len(tasks), desc="Converting"):
            total_imgs += imgs_done
            total_potholes += potholes_done
            
    print(f"\nSuccessfully converted {total_imgs} images with {total_potholes} pothole annotations!")
    
    # Generate data.yaml
    yaml_content = {
        'path': str(out_path.resolve()).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['pothole']
    }
    
    yaml_file = out_path / "data.yaml"
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Generated dataset configuration: {yaml_file}")
    
    # Also save a local relative data.yaml for portability
    rel_yaml = {
        'path': './dataset',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['pothole']
    }
    with open(out_path.parent / "data.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(rel_yaml, f, sort_keys=False)
    print(f"Generated root configuration: {out_path.parent / 'data.yaml'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert Pothole Segmentation Dataset to YOLO Format")
    parser.add_argument('--src', type=str, default=r"C:\Users\Bisman Singh\Downloads\archive (4)\data_processed", help="Path to raw data_processed directory")
    parser.add_argument('--out', type=str, default=r"./dataset", help="Output path for YOLO dataset")
    parser.add_argument('--size', type=int, default=640, help="Target image size (default: 640)")
    parser.add_argument('--min-area', type=int, default=15, help="Minimum contour pixel area threshold to filter noise")
    parser.add_argument('--workers', type=int, default=8, help="Number of concurrent workers")
    args = parser.parse_args()
    
    convert_dataset(args.src, args.out, target_size=args.size, min_area=args.min_area, num_workers=args.workers)
