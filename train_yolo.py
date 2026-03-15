import os
import subprocess

# Configuration
YOLOV5_DIR = "/home/ayush-srivastava/Desktop/Machli_project/yolov5"
PYTHON_BIN = "/home/ayush-srivastava/Desktop/Machli_project/yolov5_env/bin/python3"
DATA_YAML = "/home/ayush-srivastava/Desktop/Machli_project/yolo_dataset/dataset.yaml"

# Raspberry Pi optimal settings
# yolov5n.pt is Nano, yolov5s.pt is Small. Small is a good balance, but Nano is extremely fast.
MODEL_WEIGHTS = "yolov5s.pt" 
IMAGE_SIZE = 416
BATCH_SIZE = 16
EPOCHS = 30 # Reduced from 50 to save time since it's a simple dataset

def train_yolo():
    print(f"Starting YOLOv5 Training with weights: {MODEL_WEIGHTS}")
    print(f"Dataset config: {DATA_YAML}")
    
    # Construct YOLOv5 train command
    # Uses the python binary from the virtual env we created
    cmd = [
        PYTHON_BIN, os.path.join(YOLOV5_DIR, "train.py"),
        "--img", str(IMAGE_SIZE),
        "--batch", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--data", DATA_YAML,
        "--weights", MODEL_WEIGHTS,
        "--project", "/home/ayush-srivastava/Desktop/Machli_project/runs",
        "--name", "fish_detection_model",
        "--cache" # Cache images for faster training
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    # Run the training process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    if rc == 0:
        print("\n\n=== Training completed successfully! ===")
        print(f"Results saved to: /home/ayush-srivastava/Desktop/Machli_project/runs/fish_detection_model")
        print("You can find the best model weights at: /home/ayush-srivastava/Desktop/Machli_project/runs/fish_detection_model/weights/best.pt")
    else:
        print(f"\n\nTraining failed with return code {rc}")

if __name__ == "__main__":
    train_yolo()
