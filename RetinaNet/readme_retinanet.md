Stanford Dogs RetinaNet Detection Demo
======================================


Author: Kayode Ajayi
Department of Electrical and Computer Engineering
Concordia University


--------------------------------------------------
PROJECT OVERVIEW
--------------------------------------------------


This project implements an end-to-end object detection pipeline using
RetinaNet with a ResNet-50 backbone to detect and classify dog breeds
from the Stanford Dogs Dataset (120 breeds).


The system includes:


- Dataset conversion from PASCAL VOC XML → COCO JSON format
- Training pipeline with logging, checkpoints, AMP, and gradient clipping
- COCO evaluation (mAP / AR)
- Training loss visualization
- Precision-Recall and confusion matrix analysis
- Interactive Streamlit demo application for real-time testing


--------------------------------------------------
MODEL DETAILS
--------------------------------------------------


Architecture:
  - Detector: RetinaNet
  - Backbone: ResNet-50 + FPN


Training Setup:
  - Epochs: 25
  - Batch size: 6
  - Optimizer: AdamW
  - Learning Rate: 1e-4
  - Mixed Precision (AMP): Enabled
  - Gradient clipping: Enabled (norm = 5.0)
  - Image resolution: Original dataset size (approx. 333 x 500)
  - Training: From scratch (no pretrained RetinaNet head)


--------------------------------------------------
DATASET
--------------------------------------------------


Stanford Dogs Dataset:


- 20,580 total images
- 120 fine-grained dog breeds
- Original annotations in XML format
- Converted to COCO standard for training/evaluation


Dataset split:


- Training set: 90%
- Validation set: 10%


--------------------------------------------------
EVALUATION RESULTS
--------------------------------------------------


After 25 training epochs:


- AP@[0.50:0.95] = 0.224
- AP50           = 0.265
- AP75           = 0.245
- AR@1           = 0.686
- AR@10          = 0.712
- AR@100         = 0.712


Interpretation:


Recall is relatively high, indicating the detector locates dogs
successfully in most images.


Average precision remains moderate due to:
- Fine-grained breed similarity
- No heavy data augmentation
- Complex classification task (120 close-looking classes)


--------------------------------------------------
PROJECT FILE STRUCTURE
--------------------------------------------------


Main scripts:


convert_to_coco.py
  → Converts XML annotations to COCO JSON format.


split_train_val.py
  → Splits the dataset into training and validation sets.


dogs_coco_dataset.py
  → PyTorch dataset loader for COCO-style detection data.


train_retinanet.py
  → Main training script with AMP, gradient clipping, logging,
    and checkpoint saving.


eval_coco.py
  → Runs official COCO evaluation on the validation set.


plot_logs.py
  → Generates training loss plots from CSV logs.


confusion & PR analysis scripts
  → Generates confusion matrices and precision–recall curves.


--------------------------------------------------
DEMO APPLICATION
--------------------------------------------------


The interactive demo is implemented in:


app.py  (Streamlit Application)


Features:


- User uploads any dog image.
- The trained RetinaNet model performs inference in real time.
- Output shows:
    - Detected bounding boxes
    - Dog breed names
    - Confidence scores
- Raw prediction results are also displayed.


--------------------------------------------------
RUNNING THE DEMO
--------------------------------------------------


1) Install dependencies:
   pip install streamlit torch torchvision pillow pycocotools


2) Place the trained model file in the same folder:
   retinanet_dogs_final.pth


3) Ensure annotations train.json exists in:
   annotations/train.json


4) Launch the app:
   python -m streamlit run retinanet_app.py


5) Open your browser at:
   http://localhost:8501


--------------------------------------------------
NOTES
--------------------------------------------------


- GPU is automatically used if available.
- If running on CPU, inference is slower.
- Confidence threshold can be adjusted inside app.py.
- Images should contain a visible dog for best detection accuracy.


--------------------------------------------------
END OF README
--------------------------------------------------



