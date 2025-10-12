# Dogbreeds YOLO Workflow

Short summary
- This repository contains a YOLOv8 workflow for the Dogbreeds dataset (preparing Pascal/VOC XML -> YOLO format, basic EDA, training/eval).
- Main scripts/notebooks:
  - yolo_workflow2.py — prepares dataset, converts XML -> YOLO labels, produces dataset YAML, trains model.
  - yolo.ipynb — interactive EDA and visualization notebook.

Dataset storage & recommended hosting
- Local (small subset): place dataset under the repo at `Dogbreeds/` with this layout:
  - Dogbreeds/
    - images/Images/<wnid-breed>/*.jpg
    - annotations/Annotation/<wnid-breed>/*.xml
- Large/official dataset: host externally (recommended) and keep only metadata & download script in repo. Suggested hosts:
  - Zenodo (DOI), Hugging Face Hub, AWS S3 / GCS, or GitHub Releases.
- If storing large files in the repo, use Git LFS:
  - git lfs install
  - add patterns to `.gitattributes` (e.g., `data/* filter=lfs diff=lfs merge=lfs -text`)

Download helper
- Add a download script (example `scripts/download_data.sh`) to fetch the full dataset and verify checksums. Keep small sample images in `data/sample/` for quick tests.

How to run the preparation & EDA
1. Ensure dataset is at `Dogbreeds/` as described above.
2. Create a Python environment and install dependencies:
   - python >= 3.8
   - pip install -r requirements.txt
   - Example packages: ultralytics, lxml, opencv-python, pandas, seaborn, scikit-learn, tqdm
3. Run preparation script to convert annotations and split dataset:
   - python yolo_workflow2.py
   - This will create `dogs_yolo_dataset/` containing:
     - Images/train, Images/val
     - labels/train, labels/val
     - dogs_dataset.yaml (train/val paths, nc, names)
4. Open and run `yolo.ipynb` to perform EDA and visualize sample annotations.

Training & evaluation (YOLOv8)
- Training (example):
  - from ultralytics import YOLO
  - model = YOLO("yolov8n.pt")
  - model.train(data="dogs_yolo_dataset/dogs_dataset.yaml", epochs=20, imgsz=640, batch=16, name="dog_breed_yolov8")
- Evaluation:
  - model.val()  # after training or load checkpoint

Notebook notes
- `yolo.ipynb` contains EDA cells that compute bbox stats and produce plots in `dogs_yolo_dataset/eda`.
- If plots are empty, run the diagnostics cell in the notebook which prints counts and reveals missing/NaN fields.

Repository tips
- Include:
  - `requirements.txt`
  - `scripts/download_data.sh` (download + checksum)
  - `data/` with a small sample subset (allowed in repo)
  - `CITATION.cff` and `LICENSE` for dataset and repo usage instructions
- For reproducible releases, upload dataset snapshots to Zenodo (mint DOI) or attach to GitHub Release.

Example `.gitattributes` (for Git LFS)
```
data/* filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
```

Contacts & citation
- Add instructions for how to cite the dataset and the model if you plan to publish results.

