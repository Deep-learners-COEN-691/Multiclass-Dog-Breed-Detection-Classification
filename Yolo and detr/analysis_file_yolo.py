#%% analysis_file.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():

    import pandas as pd
    import matplotlib.pyplot as plt


    from ultralytics import YOLO

    # Load best model from training
    model = YOLO("runs/detect/dog_breed_yolo11n5/weights/best.pt")

    # Validate on the test split specified in yolo.yaml
    metrics = model.val(data="yolo_dataset/yolo.yaml", split="test")

    # Get the scalar values you need
    res = metrics.results_dict
    print(res)
    print("Precision:",     res["metrics/precision(B)"])
    print("Recall:",        res["metrics/recall(B)"])
    print("mAP50:",         res["metrics/mAP50(B)"])
    print("mAP50:95:",      res["metrics/mAP50-95(B)"])


    #%%
  

if __name__ == "__main__":
    main()











#%%

# Adjust these paths if your run names are slightly different
# yolo_results = pd.read_csv("runs/detect/dog_breed_yolo11n/results.csv")
# rtdetr_results = pd.read_csv("runs/detect/dog_breed_rtdetr_l/results.csv")

# plt.figure(figsize=(8, 5))
# plt.plot(yolo_results["epoch"], yolo_results["metrics/mAP50-95(B)"],
#          label="YOLOv11n")
# plt.plot(rtdetr_results["epoch"], rtdetr_results["metrics/mAP50-95(B)"],
#          label="RT-DETR-L")

# plt.xlabel("Epoch")
# plt.ylabel("mAP50-95 (val)")
# plt.title("Validation mAP50-95 over Epochs")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.4)
# plt.tight_layout()
# plt.savefig("map_curves_yolo_rtdetr.pdf", dpi=300)
# plt.show()
