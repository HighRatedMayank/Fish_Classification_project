import os
import shutil
import numpy as np
import cv2
from pathlib import Path
import random

# Configuration
SOURCE_DIR = "/home/ayush-srivastava/Desktop/Machli_project/archive/Fish_Dataset/Fish_Dataset"
OUTPUT_DIR = "/home/ayush-srivastava/Desktop/Machli_project/yolo_dataset"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

# 9 Fish Species (corresponding exactly to the dataset folders)
SPECIES = [
    "Black Sea Sprat", 
    "Gilt-Head Bream", 
    "Hourse Mackerel", 
    "Red Mullet", 
    "Red Sea Bream", 
    "Sea Bass", 
    "Shrimp", 
    "Striped Red Mullet", 
    "Trout"
]

def create_dirs(base_dir):
    """Creates YOLOv5 directory structure."""
    dirs = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val",
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def mask_to_yolo_bbox(mask_path):
    """
    Reads a binary mask and extracts the bounding box in YOLO format.
    Returns: (x_center, y_center, width, height) normalized [0, 1]
    """
    try:
        # Read as grayscale
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Any pixel > 128 is considered part of the mask
        binary = img > 128
            
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None # Empty mask

        # Find bounding box pixel coordinates
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        h, w = img.shape[:2]
        
        # Convert to YOLO (normalized)
        x_center = (cmin + cmax) / 2.0 / w
        y_center = (rmin + rmax) / 2.0 / h
        bbox_w = (cmax - cmin) / float(w)
        bbox_h = (rmax - rmin) / float(h)
        
        return (x_center, y_center, bbox_w, bbox_h)
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return None

def process_dataset():
    print("Starting dataset preparation...")
    create_dirs(OUTPUT_DIR)
    
    total_processed = 0
    total_skipped = 0
    
    for class_id, species in enumerate(SPECIES):
        species_dir = os.path.join(SOURCE_DIR, species, species)
        gt_dir = os.path.join(SOURCE_DIR, species, f"{species} GT")
        
        if not os.path.exists(species_dir) or not os.path.exists(gt_dir):
            print(f"Skipping {species}: Directory missing")
            continue
            
        images = [f for f in os.listdir(species_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(images)} images for {species} (ID: {class_id})")
        
        # Shuffle for random train/val split
        random.seed(42)
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_files = images[:split_idx]
        val_files = images[split_idx:]
        
        for file_list, split_name in [(train_files, 'train'), (val_files, 'val')]:
            for img_file in file_list:
                img_path = os.path.join(species_dir, img_file)
                mask_path = os.path.join(gt_dir, img_file)
                
                if not os.path.exists(mask_path):
                    total_skipped += 1
                    continue
                    
                bbox = mask_to_yolo_bbox(mask_path)
                if bbox is None:
                    total_skipped += 1
                    continue
                    
                # New paths
                base_name = f"{species.replace(' ', '_')}_{img_file.split('.')[0]}"
                dest_img_path = os.path.join(OUTPUT_DIR, "images", split_name, f"{base_name}.png")
                dest_label_path = os.path.join(OUTPUT_DIR, "labels", split_name, f"{base_name}.txt")
                
                # Copy image
                shutil.copy2(img_path, dest_img_path)
                
                # Write YOLO label
                with open(dest_label_path, 'w') as f:
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                    
                total_processed += 1
                if total_processed % 1000 == 0:
                    print(f"Processed {total_processed} items...")

    # Write dataset.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

nc: {len(SPECIES)}
names:
"""
    for sp in SPECIES:
        yaml_content += f"  - '{sp}'\n"
        
    with open(os.path.join(OUTPUT_DIR, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"\nDone! Processed: {total_processed}, Skipped: {total_skipped}")
    print(f"YOLO dataset is ready at: {OUTPUT_DIR}")
    print(f"Config generated at: {os.path.join(OUTPUT_DIR, 'dataset.yaml')}")

if __name__ == "__main__":
    process_dataset()
