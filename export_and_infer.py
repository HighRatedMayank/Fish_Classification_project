import os
import subprocess
import cv2

YOLOV5_DIR = "/home/ayush-srivastava/Desktop/Machli_project/yolov5"
PYTHON_BIN = "/home/ayush-srivastava/Desktop/Machli_project/yolov5_env/bin/python3"
# Once training is fully complete, the best weights will be here:
WEIGHTS_PATH = "/home/ayush-srivastava/Desktop/Machli_project/runs/fish_detection_model/weights/best.pt"

def export_model_for_pi():
    """
    Exports the trained PyTorch model to NCNN and ONNX formats.
    NCNN is highly optimized for ARM processors like the Raspberry Pi.
    """
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Trained weights not found at {WEIGHTS_PATH}")
        print("Please ensure you have completed training first.")
        return

    print("Exporting model to NCNN (for Raspberry Pi)...")
    export_cmd = [
        PYTHON_BIN, os.path.join(YOLOV5_DIR, "export.py"),
        "--weights", WEIGHTS_PATH,
        "--include", "ncnn", "onnx",
        "--img", "416"
    ]
    
    subprocess.run(export_cmd)
    print("Export complete! You will find the .ncnn and .onnx files in the weights directory.")

def test_inference(image_path):
    """
    Runs a test inference using the exported models.
    """
    if not os.path.exists(image_path):
        print("Image not found.")
        return
        
    print(f"Running inference on {image_path}...")
    infer_cmd = [
        PYTHON_BIN, os.path.join(YOLOV5_DIR, "detect.py"),
        "--weights", WEIGHTS_PATH, 
        "--source", image_path,
        "--img", "416",
        "--conf", "0.25"
    ]
    subprocess.run(infer_cmd)
    
if __name__ == "__main__":
    print("=== Raspberry Pi Deployment Preparation ===")
    print("1. Exporting...")
    export_model_for_pi()
    
    # Example inference if weights existed:
    # test_inference("/home/ayush-srivastava/Desktop/Machli_project/yolo_dataset/images/val/Trout_00001.png")
