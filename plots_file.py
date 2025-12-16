
#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
print(torch.__version__)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))



#%%

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load results
# yolo_results = pd.read_csv("C:\\Users\\eroew\\Downloads\\Fall_Love_Assignmets\\object_detection\\dogbreed_computer_vision_project\\report_images\\results_full_yolo.csv")
# rtdetr_results = pd.read_csv("C:\\Users\\eroew\\Downloads\\Fall_Love_Assignmets\\object_detection\\dogbreed_computer_vision_project\\report_images\\results_full_retr.csv")

# # Create subplots: 1 row, 2 columns
# fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# # --- YOLO subplot ---
# axes[0].plot(yolo_results["epoch"], yolo_results["metrics/mAP50-95(B)"])
# axes[0].set_title("YOLOv11n")
# axes[0].set_xlabel("Epoch")
# axes[0].set_ylabel("mAP50-95 (val)")
# axes[0].grid(True)

# # --- RT-DETR subplot ---
# axes[1].plot(rtdetr_results["epoch"], rtdetr_results["metrics/mAP50-95(B)"])
# axes[1].set_title("RT-DETR-L")
# axes[1].set_xlabel("Epoch")
# axes[1].grid(True)

# fig.suptitle("Validation mAP50-95 over Epochs")
# plt.tight_layout()
# plt.savefig("map_curves_yolo_rtdetr_subplots.png", dpi=300)
# plt.show()

# %% inference time
import time
from ultralytics import YOLO, RTDETR
from pathlib import Path
import glob

# Load trained models
yolo_model = YOLO("runs/detect/dog_breed_yolo11n5/weights/best.pt")
rtdetr_model = RTDETR("runs/detect/dog_breed_rtdetr_l222/weights/best.pt")

test_images = sorted(glob.glob("yolo_dataset/test/images/*.jpg"))[:200]  # subset for speed

def benchmark(model, imgs):
    start = time.time()
    for img in imgs:
        model.predict(img, verbose=False)
    end = time.time()
    total_time = end - start
    avg_time = total_time / len(imgs)
    fps = 1.0 / avg_time
    return avg_time * 1000.0, fps  # ms, FPS

yolo_ms, yolo_fps = benchmark(yolo_model, test_images)
rtdetr_ms, rtdetr_fps = benchmark(rtdetr_model, test_images)

print("YOLOv11n: {:.2f} ms / image, {:.2f} FPS".format(yolo_ms, yolo_fps))
print("RT-DETR-L: {:.2f} ms / image, {:.2f} FPS".format(rtdetr_ms, rtdetr_fps))


# %%
