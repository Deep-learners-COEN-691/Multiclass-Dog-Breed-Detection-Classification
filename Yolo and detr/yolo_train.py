import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from pathlib import Path

def main():
    
    
    
    

    # Path to your dataset root (same as in the YAML creation code)
    DATASET_ROOT = Path("./yolo_dataset")
    DATA_CONFIG = DATASET_ROOT / "yolo.yaml"

    # Load a small pretrained model (change to yolo11s.pt / yolo11m.pt later if you want)
    # If you're on YOLOv8 instead of YOLOv11, use "yolov8n.pt" here.
    
    
    
    model = YOLO("yolo11n.pt")
    
    last_weights_path = "./runs/detect/dog_breed_yolo11n5/weights/last.pt"
    if os.path.exists(last_weights_path):
        print(f"Loading weights from {last_weights_path}")
        model = YOLO(last_weights_path)
    

    # Recommended-style training settings
    results = model.train(
        data=str(DATA_CONFIG),     # path to the yolo.yaml file
        epochs=10,                # 100 is a common starting point; reduce if just testing
        imgsz=640,                 # default image size
        batch=4,                  # auto batch size based on your VRAM / memory
        device=0,              # Apple M1/M2 GPU; use "0" for CUDA, "cpu" otherwise
        name="dog_breed_yolo11n",  # experiment name (controls runs/ folder)
        workers=8,                 # number of dataloader workers (tweak if you get issues)
        pretrained=True,           # use pretrained weights (recommended)
        patience=50,               # early stopping if val loss doesn't improve
    )

if __name__ == "__main__":
    main()