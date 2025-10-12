# Dogbreeds — YOLOv8 Workflow

Project overview
- Prepares the Dogbreeds Pascal‑VOC XML annotations into YOLO format, performs dataset EDA, trains a YOLOv8 model, and evaluates results.
- Key artifacts:
  - Prepared dataset: `dogs_yolo_dataset/`
  - EDA outputs: `dogs_yolo_dataset/eda/` (plots, CSVs, annotated samples)
  - Training outputs: `runs/` (Ultralytics default)

Prerequisites
- Python 3.8+ (recommended 3.9)
- Install dependencies:
  - pip install -r requirements.txt
  - (or) pip install ultralytics lxml opencv-python pandas seaborn scikit-learn tqdm matplotlib

Repository layout (important files)
- yolo_workflow2.py — dataset prep (XML → YOLO), split, YAML generation, optional training
- yolo.ipynb — interactive EDA and visualization (saves plots to eda/)
- dogs_yolo_dataset/ — generated dataset (Images/, labels/, dogs_dataset.yaml)
- runs/ — model training outputs (checkpoints, logs)
- README.md — this file
- scripts/download_data.sh (optional) — helper to fetch large dataset externally

Dataset expected layout
Place the dataset at repository root as:
```
Dogbreeds/
  images/Images/<wnid-breed>/*.jpg
  annotations/Annotation/<wnid-breed>/*.xml
```
If the dataset is large, host externally (Zenodo, S3, HF) and include only a small sample in the repo plus a download script. Use Git LFS for medium files.

Quick start — prepare dataset and run EDA
1. Ensure dataset is present under `Dogbreeds/` (see layout above).
2. Run preparation script to convert annotations and split:
   - python yolo_workflow2.py
   - Output: `dogs_yolo_dataset/Images/train`, `dogs_yolo_dataset/Images/val`, `dogs_yolo_dataset/labels/train`, `dogs_yolo_dataset/labels/val`, and `dogs_yolo_dataset/dogs_dataset.yaml`
3. Open the notebook and run cells (or run EDA cells from CLI):
   - jupyter notebook yolo.ipynb
   - EDA outputs saved under `dogs_yolo_dataset/eda/`

Where outputs are saved (paths)
- Converted dataset & YAML:
  - dogs_yolo_dataset/Images/{train,val}/*.jpg
  - dogs_yolo_dataset/labels/{train,val}/*.txt
  - dogs_yolo_dataset/dogs_dataset.yaml
- EDA artifacts:
  - dogs_yolo_dataset/eda/class_frequency.csv
  - dogs_yolo_dataset/eda/objects_per_image_hist.png
  - dogs_yolo_dataset/eda/class_frequency_top.png
  - dogs_yolo_dataset/eda/bbox_area_norm_hist.png
  - dogs_yolo_dataset/eda/bbox_aspect_ratio_hist.png
  - dogs_yolo_dataset/eda/image_size_scatter.png
  - dogs_yolo_dataset/eda/sample_<breed>.jpg (annotated image samples)
- Training runs: `runs/detect/<exp_name>/` (best.pt, results.json, training logs)

EDA — what each plot reveals and what to check
- class_frequency.csv / class_frequency_top.png
  - Shows instance counts per class. Check for severe class imbalance (long tail).
- objects_per_image_hist.png
  - Shows how many objects per image. Look for many zeros or extreme counts.
- bbox_area_norm_hist.png
  - BBox area normalized by image area. If most boxes are extremely small, training may struggle.
- bbox_aspect_ratio_hist.png
  - Width/height distribution. Long tails or values near zero indicate mislabeled boxes or incorrect coordinates.
- image_size_scatter.png
  - Shows diversity of image dimensions; many tiny or identical sizes may require resizing strategy.
- sample_<breed>.jpg
  - Inspect sample annotated images to confirm box alignment and labels.

Example inline references (view after running EDA)
- ![Top classes](dogs_yolo_dataset/eda/class_frequency_top.png)
- ![Aspect ratios](dogs_yolo_dataset/eda/bbox_aspect_ratio_hist.png)
- ![Sample annotate](dogs_yolo_dataset/eda/sample_Chihuahua.jpg)

Training & evaluation (YOLOv8)
- Example training snippet (in Python):
```py
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="dogs_yolo_dataset/dogs_dataset.yaml", epochs=20, imgsz=640, batch=16, name="dog_breed_yolov8")
```
- Evaluate:
```py
results = model.val()  # runs evaluation on val set defined in YAML
```
- Check `runs/detect/<exp_name>/` for `best.pt`, `results.json`, and training curves.

Tips & common troubleshooting
- Empty plots: run the diagnostics cell in `yolo.ipynb` to inspect `df` (counts, NaNs). Verify parse_xml_coordiantes returns valid numbers and class mapping covers XML name variants.
- Missing images or XMLs: verify folder names match expected `wnid-breed` format, and .DS_Store or hidden files are not interfering.
- Use Git LFS for medium-sized dataset files and keep only a small sample in the repo.
- For reproducible dataset snapshots, use releases + Zenodo (mint DOI) or HF dataset hosting.

Versioning & dataset hosting recommendations
- Small sample: include `data/sample/` in repo.
- Full dataset: host externally and include `scripts/download_data.sh` + `checksum.sha256` in the repo.
- Use GitHub Releases or Zenodo to version dataset snapshots and provide a DOI.

License & citation
- Add LICENSE and CITATION.cff to the repo with dataset license and citation instructions.

Contact
- Open an issue with notebook outputs or error logs for reproducibility help.

