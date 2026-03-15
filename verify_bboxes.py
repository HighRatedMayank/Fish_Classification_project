import os
import cv2
import random

YOLO_DIR = "/home/ayush-srivastava/Desktop/Machli_project/yolo_dataset"
OUTPUT_DIR = "/home/ayush-srivastava/Desktop/Machli_project/validation_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read classes
with open(os.path.join(YOLO_DIR, 'dataset.yaml'), 'r') as f:
    lines = f.readlines()
    
# Get some train images
train_imgs = os.listdir(os.path.join(YOLO_DIR, 'images/train'))
samples = random.sample(train_imgs, min(20, len(train_imgs)))

print(f"Generating {len(samples)} validation images with bounding boxes...")

for img_name in samples:
    img_path = os.path.join(YOLO_DIR, 'images/train', img_name)
    lbl_path = os.path.join(YOLO_DIR, 'labels/train', img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
    
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                class_id, x_c, y_c, bw, bh = map(float, line.strip().split())
                
                # Convert back to pixel coords
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Class {int(class_id)}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img)

print(f"Validation images saved to: {OUTPUT_DIR}")
