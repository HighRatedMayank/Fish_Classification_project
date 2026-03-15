# Machli: Fish Breed Detection Project 🐟

This repository contains the complete pipeline for preparing, training, and deploying a YOLOv5 object detection model to an edge device (like a Raspberry Pi) to detect 9 different species of fish.

## 🚀 Accomplished So Far

### 1. Automated Dataset Preparation & Bounding Box Generation
Instead of manually drawing 9,000 bounding boxes by hand, the dataset provides pixel-perfect segmentation masks. We created a script (`dataset_prep.py`) that successfully processed the 9,000 images and:
- Read every segmentation mask (the silhouette of the fish).
- Extracted the tightest perfectly accurate X, Y coordinates for a YOLO bounding box.
- Generated the YOLO `.txt` label files for all images.
- Automatically split and organized the data into `train` (80%) and `val` (20%) datasets.
- Created the required YOLO configuration file (`yolo_dataset/dataset.yaml`).

### 2. Environment Setup
- A clean Python virtual environment (`yolov5_env`) has been successfully created.
- The official YOLOv5 repository has been cloned to the root of this project.
- All dependencies (`torch`, `ultralytics`, `pandas`, `opencv`, etc.) have been installed.

### 3. Pipeline Scripts Prepared
The following scripts have been written and are ready for execution:
- `train_yolo.py`: A wrapper script that properly launches the YOLOv5 training loop using the custom dataset and the `yolov5s.pt` (Small) pre-trained weights, which provide the best balance of speed and accuracy for a Raspberry Pi.
- `export_and_infer.py`: The final deployment script that will export the PyTorch model (`best.pt`) to NCNN and ONNX formats—both of which are highly optimized for inference on ARM-based edge devices like the Raspberry Pi.

---

## ⏳ Next Steps for the Next Developer

The project is fully ready for the model training and deployment phases.

### Step 1: Train the Model
The environment and dataset are ready. To begin training the YOLOv5 model on the dataset, run:
```bash
source yolov5_env/bin/activate
python3 train_yolo.py
```
> **Note:** Training 9,000 images on a standard CPU could take several hours. If a GPU (CUDA) is available, training will be substantially faster. The resulting best model weights will automatically save to `runs/fish_detection_model/weights/best.pt`.

### Step 2: Export for Raspberry Pi
PyTorch `.pt` models are heavy and slow on Raspberry Pis. Once training is fully complete, convert the model to lightweight NCNN format by running:
```bash
source yolov5_env/bin/activate
python3 export_and_infer.py
```
This will generate `.ncnn` and `.onnx` files in the `weights/` directory.

### Step 3: Deployment
Copy the exported `.ncnn` model file, along with your inference script, to the Raspberry Pi. The Raspberry Pi can load the NCNN model and run real-time inference using a connected camera module.
