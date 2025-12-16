import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
# import torch
from ultralytics import RTDETR


def main():
    # ----------------- DATASET CONFIG -----------------
    # Same dataset structure you used for YOLO:
    # yolo_dataset/
    #   ├─ train/images, train/labels
    #   ├─ val/images,   val/labels
    #   ├─ test/images,  test/labels
    #   └─ yolo.yaml
    DATASET_ROOT = Path("./yolo_dataset")
    DATA_CONFIG = DATASET_ROOT / "yolo.yaml"
    
    #check if cuda is available
    # if torch.cuda.is_available():
    #     print("CUDA is available. Using GPU for training.")
    # else:
    #     print("CUDA is not available. Using CPU for training.")
       

    # Experiment name (controls runs/detect/<name>/)
    EXP_NAME = "dog_breed_rtdetr_l22"

    # ----------------- MODEL LOADING -----------------
    # Default COCO-pretrained RT-DETR model
    pretrained_ckpt = "rtdetr-l.pt"  # or "rtdetr-x.pt" if you want the bigger one

    # If you've already trained once, resume from last weights
    last_weights_path = Path("runs") / "detect" / EXP_NAME / "weights" / "last.pt"

    if last_weights_path.exists():
        print(f"Loading fine-tuned weights from: {last_weights_path}")
        model = RTDETR(str(last_weights_path))
    else:
        print(f"Loading COCO-pretrained RT-DETR model: {pretrained_ckpt}")
        model = RTDETR(pretrained_ckpt)

    # ----------------- TRAINING -----------------
    results = model.train(
        data=str(DATA_CONFIG),   # path to your yolo.yaml
        epochs=85,               # adjust as needed
        imgsz=640,
        batch=4,                 # tune based on GPU VRAM
        device=0,                # 0 for first CUDA GPU, "cpu" for CPU
        name=EXP_NAME,           # experiment name
        workers=0,
        patience=50,             # early stopping
        # other useful options if you want:
        # lr0=0.01,
        # lrf=0.01,
        # mosaic=1.0,
    )

    # Optional: save a nicely named final checkpoint
    model.save(f"{EXP_NAME}_final.pt")
    print("Training complete. Final weights saved as:", f"{EXP_NAME}_final.pt")


if __name__ == "__main__":
    main()
