import os
import sys
import zipfile
import xml.etree.ElementTree as ET
import random
from pathlib import Path
import cv2
import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def convert_voc_to_yolo(xml_content, img_width, img_height):
    root = ET.fromstring(xml_content)
    yolo_lines = []
    
    size_elem = root.find("size")
    if size_elem is not None:
        try:
            xml_w = float(size_elem.find("width").text)
            xml_h = float(size_elem.find("height").text)
            if xml_w > 0 and xml_h > 0:
                img_width = xml_w
                img_height = xml_h
        except (AttributeError, ValueError):
            pass
            
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None:
            continue
        name = name_elem.text.strip().lower()
        if "pothole" not in name:
            continue
            
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
            
        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (AttributeError, ValueError):
            continue
            
        xmin = max(0.0, min(float(img_width), xmin))
        ymin = max(0.0, min(float(img_height), ymin))
        xmax = max(0.0, min(float(img_width), xmax))
        ymax = max(0.0, min(float(img_height), ymax))
        
        bw = xmax - xmin
        bh = ymax - ymin
        
        if bw <= 1 or bh <= 1:
            continue
            
        x_center = (xmin + bw / 2.0) / float(img_width)
        y_center = (ymin + bh / 2.0) / float(img_height)
        w_norm = bw / float(img_width)
        h_norm = bh / float(img_height)
        
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))
        
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
    return yolo_lines

def ingest_dataset(
    zip_path=r"C:\Users\Bisman Singh\Downloads\archive (3).zip",
    dataset_dir="dataset",
    train_ratio=0.80,
    val_ratio=0.10,
    test_ratio=0.10,
    seed=42
):
    print("=" * 65)
    print("  🚀 Ingesting & Integrating VOC Pothole Dataset into YOLO Model")
    print("=" * 65)
    
    zip_p = Path(zip_path).resolve()
    if not zip_p.exists():
        raise FileNotFoundError(f"Source archive not found: {zip_p}")
        
    base_dir = Path(dataset_dir).resolve()
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        
    print(f"📦 Source archive: {zip_p}")
    print(f"📁 Target dataset directory: {base_dir}")
    
    with zipfile.ZipFile(zip_p, 'r') as zf:
        namelist = zf.namelist()
        xml_files = sorted([f for f in namelist if f.startswith("annotations/") and f.endswith(".xml")])
        print(f"🔎 Found {len(xml_files)} annotation files in archive.")
        
        pairs = []
        for xml_file in xml_files:
            stem = Path(xml_file).stem
            possible_imgs = [
                f"images/{stem}.png",
                f"images/{stem}.jpg",
                f"images/{stem}.jpeg"
            ]
            found_img = None
            for pimg in possible_imgs:
                if pimg in namelist:
                    found_img = pimg
                    break
            if found_img:
                pairs.append((xml_file, found_img, stem))
                
        print(f"🔗 Successfully matched {len(pairs)} image-annotation pairs.")
        
        random.seed(seed)
        random.shuffle(pairs)
        
        n_total = len(pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:]
        
        splits_map = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs
        }
        
        stats = {"train": 0, "val": 0, "test": 0, "boxes": 0}
        
        for split_name, split_list in splits_map.items():
            print(f"\nProcessing {split_name} split ({len(split_list)} samples)...")
            for xml_file, img_file, stem in split_list:
                img_bytes = zf.read(img_file)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"⚠️ Warning: Could not decode {img_file}")
                    continue
                    
                h, w = img.shape[:2]
                xml_content = zf.read(xml_file).decode("utf-8", errors="replace")
                yolo_labels = convert_voc_to_yolo(xml_content, w, h)
                
                target_stem = f"voc_{stem}"
                out_img_path = images_dir / split_name / f"{target_stem}.jpg"
                out_lbl_path = labels_dir / split_name / f"{target_stem}.txt"
                
                cv2.imwrite(str(out_img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                with open(out_lbl_path, "w", encoding="utf-8") as lf:
                    if yolo_labels:
                        lf.write("\n".join(yolo_labels) + "\n")
                        stats["boxes"] += len(yolo_labels)
                    else:
                        lf.write("")
                        
                stats[split_name] += 1
                
        print("\n" + "=" * 65)
        print("  🎉 VOC Dataset Ingestion Completed Successfully!")
        print("=" * 65)
        print(f"Added to Train: {stats['train']} images")
        print(f"Added to Val:   {stats['val']} images")
        print(f"Added to Test:  {stats['test']} images")
        print(f"Total New Pothole Bounding Boxes: {stats['boxes']}")
        
        for cache_file in labels_dir.glob("*.cache"):
            try:
                cache_file.unlink()
                print(f"🗑️ Removed stale cache: {cache_file.name}")
            except Exception as e:
                print(f"⚠️ Could not delete cache {cache_file}: {e}")
                
    return stats

if __name__ == "__main__":
    ingest_dataset()
