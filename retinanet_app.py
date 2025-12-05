# ============================================================
# Streamlit Demo App for Stanford Dogs RetinaNet
# ------------------------------------------------------------
# This app provides a lightweight interactive interface
# to test the trained RetinaNet dog-breed detector.
#
# Users can upload an image and visualize:
#   - Bounding boxes
#   - Predicted breed labels
#   - Confidence scores
#
# The demo uses the trained RetinaNet + ResNet50 model.
# ============================================================

import streamlit as st
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models import ResNet50_Weights # RetinaNet_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont
import json
import os
import numpy as np

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

MODEL_PATH = "retinanet_dogs_final.pth"    # <-- your trained model path
COCO_JSON  = "annotations/train.json"  # used to load breed names
CONF_THRESH = 0.30
IOU_THRESH  = 0.40
FONT_SIZE = 18

# ------------------------------------------------------------
# LOAD CLASS LABELS
# ------------------------------------------------------------

def load_class_names(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = coco["categories"]
    cats = sorted(cats, key=lambda x: x["id"])
    names = {c["id"]: c["name"] for c in cats}

    return names

CLASS_NAMES = load_class_names(COCO_JSON)

NUM_CLASSES = len(CLASS_NAMES) + 1

# ------------------------------------------------------------
# MODEL LOADER
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=NUM_CLASSES,
    )

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt)

    model.to(device).eval()
    return model, device

# ------------------------------------------------------------
# DRAW RESULTS
# ------------------------------------------------------------

def draw_predictions(img, outputs):

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    boxes  = outputs["boxes"]
    scores = outputs["scores"]
    labels = outputs["labels"]

    for box, score, label in zip(boxes, scores, labels):
        score = score.item()

        if score < CONF_THRESH:
            continue

        x1,y1,x2,y2 = box.tolist()
        name = CLASS_NAMES.get(int(label.item()), "Unknown")

        # ------------------------
        # Bounding box
        # ------------------------
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=(255, 0, 0),
            width=3
        )

        text = f"{name} : {score:.2f}"

        # ------------------------
        # Text background
        # ------------------------

        text_w, text_h = draw.textbbox((0,0), text, font=font)[2:]
        pad = 4

        bg = [
            x1,
            max(0, y1 - text_h - pad*2),
            x1 + text_w + pad*2,
            y1
        ]

        draw.rectangle(bg, fill=(0,0,0))

        txt_pos = (bg[0] + pad, bg[1] + pad)

        # ------------------------
        # Draw text
        # ------------------------
        draw.text(
            txt_pos,
            text,
            font=font,
            fill=(255,255,255)
        )

    return img


# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

@torch.no_grad()
def run_inference(model, device, image):

    img_tensor = F.to_tensor(image).to(device)

    outputs = model([img_tensor])[0]

    return outputs

# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

st.set_page_config(page_title="Stanford Dogs RetinaNet Demo",
                   page_icon="🐶",
                   layout="wide")

st.title("🐶 Stanford Dogs RetinaNet Demo")
st.markdown(
    """
    **Upload a dog image to detect & classify its breed**

    - Model: RetinaNet + ResNet50  
    - Training: 25 epochs from scratch  
    - Dataset: Stanford Dogs (120 breeds)  
    - Metrics: AP=0.224 · AR=0.712  
    """
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded is not None:

    colA, colB = st.columns(2)

    image = Image.open(uploaded).convert("RGB")

    colA.subheader("Input")
    colA.image(image, use_column_width=True)

    model, device = load_model()

    with st.spinner("Running detection..."):
        outputs = run_inference(model, device, image)

    result_img = image.copy()
    result_img = draw_predictions(result_img, outputs)

    colB.subheader("Prediction")
    colB.image(result_img, use_column_width=True)

    st.markdown("---")

    st.subheader("Raw Predictions")

    data = []

    for b,s,l in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if s.item() < CONF_THRESH: 
            continue

        data.append({
            "Breed": CLASS_NAMES.get(int(l.item()), "Unknown"),
            "Confidence": round(float(s.item()), 3),
            "BBox": [round(v,1) for v in b.tolist()]
        })

    st.json(data)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")
st.caption(
    "Kayode Ajayi | Concordia University | ELEC — RetinaNet Project"
)
