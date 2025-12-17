#%% analysis_file.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import RTDETR

def main():
    # 1. Load trained RT-DETR model
    model = RTDETR("runs/detect/dog_breed_rtdetr_l222/weights/best.pt")

    # 2. Run validation on CPU to avoid device mismatch
    metrics = model.val(
        data="yolo_dataset/yolo.yaml",
        split="test",
        device=0,   
        workers=0,      
        batch=4         
    )
    
    # 3. Extract metrics
    res = metrics.results_dict

    print("All metrics:")
    for k, v in res.items():
        print(f"{k}: {v}")

    # Try both possible key names (Ultralytics versions differ a bit)
    mAP50    = res.get("metrics/mAP50(B)",     res.get("metrics/mAP50"))
    mAP5095  = res.get("metrics/mAP50-95(B)",  res.get("metrics/mAP50-95"))
    precision = res.get("metrics/precision(B)", res.get("metrics/precision"))
    recall    = res.get("metrics/recall(B)",    res.get("metrics/recall"))

    print("\n--- RT-DETR-L Summary ---")
    print("Precision:", precision)
    print("Recall:   ", recall)
    print("mAP50:    ", mAP50)
    print("mAP50-95: ", mAP5095)

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
