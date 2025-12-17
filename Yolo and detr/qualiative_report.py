# %% imports
import time
import glob
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, RTDETR

# ==========================
# 1. Load trained models
# ==========================
yolo_model = YOLO("runs/detect/dog_breed_yolo11n5/weights/best.pt")
rtdetr_model = RTDETR("runs/detect/dog_breed_rtdetr_l222/weights/best.pt")

# If you have a list of class names (from your yolo.yaml), put it here.
# Otherwise, labels will be just class indices.
CLASS_NAMES = [
    "affenpinscher",
    "afghan_hound",
    "african_hunting_dog",
    "airedale",
    "american_staffordshire_terrier",
    "appenzeller",
    "australian_terrier",
    "basenji",
    "basset",
    "beagle",
    "bedlington_terrier",
    "bernese_mountain_dog",
    "black-and-tan_coonhound",
    "blenheim_spaniel",
    "bloodhound",
    "bluetick",
    "border_collie",
    "border_terrier",
    "borzoi",
    "boston_bull",
    "bouvier_des_flandres",
    "boxer",
    "brabancon_griffon",
    "briard",
    "brittany_spaniel",
    "bull_mastiff",
    "cairn",
    "cardigan",
    "chesapeake_bay_retriever",
    "chihuahua",
    "chow",
    "clumber",
    "cocker_spaniel",
    "collie",
    "curly-coated_retriever",
    "dandie_dinmont",
    "dhole",
    "dingo",
    "doberman",
    "english_foxhound",
    "english_setter",
    "english_springer",
    "entlebucher",
    "eskimo_dog",
    "flat-coated_retriever",
    "french_bulldog",
    "german_shepherd",
    "german_short-haired_pointer",
    "giant_schnauzer",
    "golden_retriever",
    "gordon_setter",
    "great_dane",
    "great_pyrenees",
    "greater_swiss_mountain_dog",
    "groenendael",
    "ibizan_hound",
    "irish_setter",
    "irish_terrier",
    "irish_water_spaniel",
    "irish_wolfhound",
    "italian_greyhound",
    "japanese_spaniel",
    "keeshond",
    "kelpie",
    "kerry_blue_terrier",
    "komondor",
    "kuvasz",
    "labrador_retriever",
    "lakeland_terrier",
    "leonberg",
    "lhasa",
    "malamute",
    "malinois",
    "maltese_dog",
    "mexican_hairless",
    "miniature_pinscher",
    "miniature_poodle",
    "miniature_schnauzer",
    "newfoundland",
    "norfolk_terrier",
    "norwegian_elkhound",
    "norwich_terrier",
    "old_english_sheepdog",
    "otterhound",
    "papillon",
    "pekinese",
    "pembroke",
    "pomeranian",
    "pug",
    "redbone",
    "rhodesian_ridgeback",
    "rottweiler",
    "saint_bernard",
    "saluki",
    "samoyed",
    "schipperke",
    "scotch_terrier",
    "scottish_deerhound",
    "sealyham_terrier",
    "shetland_sheepdog",
    "shih-tzu",
    "siberian_husky",
    "silky_terrier",
    "soft-coated_wheaten_terrier",
    "staffordshire_bullterrier",
    "standard_poodle",
    "standard_schnauzer",
    "sussex_spaniel",
    "tibetan_mastiff",
    "tibetan_terrier",
    "toy_poodle",
    "toy_terrier",
    "vizsla",
    "walker_hound",
    "weimaraner",
    "welsh_springer_spaniel",
    "west_highland_white_terrier",
    "whippet",
    "wire-haired_fox_terrier",
    "yorkshire_terrier",
]



# ==========================
# 2. Benchmark inference time (your original code)
# ==========================
def benchmark(model, imgs):
    start = time.time()
    for img in imgs:
        model.predict(img, verbose=False)
    end = time.time()
    total_time = end - start
    avg_time = total_time / len(imgs)
    fps = 1.0 / avg_time
    return avg_time * 1000.0, fps  # ms, FPS


# test_images = sorted(glob.glob("yolo_dataset/test/images/*.jpg"))[:200]

# yolo_ms, yolo_fps = benchmark(yolo_model, test_images)
# rtdetr_ms, rtdetr_fps = benchmark(rtdetr_model, test_images)

# print("YOLOv11n: {:.2f} ms / image, {:.2f} FPS".format(yolo_ms, yolo_fps))
# print("RT-DETR-L: {:.2f} ms / image, {:.2f} FPS".format(rtdetr_ms, rtdetr_fps))


# ==========================
# 3. Helper functions
# ==========================
def load_yolo_labels(label_path, img_w, img_h):
    """
    Load YOLO-format labels (class x_c y_c w h) and convert to pixel xyxy.
    Returns: list of (cls_id, x1, y1, x2, y2)
    """
    boxes = []
    if not Path(label_path).is_file():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
            x1 = (xc - w / 2.0) * img_w
            y1 = (yc - h / 2.0) * img_h
            x2 = (xc + w / 2.0) * img_w
            y2 = (yc + h / 2.0) * img_h
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_boxes(image, boxes, color, class_names=None, thickness=2):
    """
    Draw boxes on a copy of the image.
    boxes: list of (cls_id, x1, y1, x2, y2, [score])
    color: (B, G, R)
    """
    img = image.copy()
    for b in boxes:
        if len(b) == 5:
            cls_id, x1, y1, x2, y2 = b
            score = None
        else:
            cls_id, x1, y1, x2, y2, score = b
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(img, pt1, pt2, color, thickness)

        # label text
        label = str(cls_id)
        if class_names is not None and 0 <= cls_id < len(class_names):
            label = class_names[cls_id]
        if score is not None:
            label = f"{label} {score:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (pt1[0], pt1[1] - th - 2), (pt1[0] + tw, pt1[1]), color, -1)
        cv2.putText(img, label, (pt1[0], pt1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def visualize_folder_with_gt(model, model_name, images_dir, labels_dir, out_dir, class_names=None):
    """
    For folders that have labels:
      - images_dir: path to images
      - labels_dir: path to YOLO .txt labels with same basename
    Saves side-by-side GT vs prediction images for every file.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground-truth boxes
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_yolo_labels(label_path, w, h)

        gt_boxes_draw = [(cls, x1, y1, x2, y2) for (cls, x1, y1, x2, y2) in gt_boxes]
        img_gt = draw_boxes(img, gt_boxes_draw, color=(0, 255, 0), class_names=class_names)

        # Predictions
        results = model.predict(str(img_path), verbose=False)[0]
        preds = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cid, sc in zip(xyxy, cls_ids, scores):
                preds.append((cid, x1, y1, x2, y2, sc))

        img_pred = draw_boxes(img, preds, color=(0, 0, 255), class_names=class_names)

        # Side-by-side: left = GT, right = prediction
        combined = np.hstack([img_gt, img_pred])

        out_path = out_dir / f"{img_path.stem}_gt_vs_pred.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"[{model_name}] Saved {out_path}")


def visualize_folder_predictions_only(model, model_name, images_dir, out_dir, class_names=None):
    """
    For folders that have only images (no labels).
    Draws prediction boxes and saves per image.
    """
    images_dir = Path(images_dir)
    out_dir = Path(out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(str(img_path), verbose=False)[0]
        preds = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cid, sc in zip(xyxy, cls_ids, scores):
                preds.append((cid, x1, y1, x2, y2, sc))

        img_pred = draw_boxes(img, preds, color=(0, 0, 255), class_names=class_names)

        out_path = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), img_pred)
        print(f"[{model_name}] Saved {out_path}")


# ==========================
# 4. Paths for your qualitative sets
# ==========================
# You can change these to whatever you are using. Example:
EASY_IMAGES_DIR = "report_images/Easy Images/images"
EASY_LABELS_DIR = "report_images/Easy Images/labels"

OCCL_IMAGES_DIR = "report_images/Occlusion and Clutter Images/images"
OCCL_LABELS_DIR = "report_images/Occlusion and Clutter Images/labels"

# For fine-grained confusion pairs, you said you only have images.
# You can put all “confusion pair” images together in one folder,
# or split them by pair. Here is one way with subfolders per pair:
P_REFINED_ROOT = Path("report_images/Fine-Grained Confusion/images")

P_POODLES_DIR = P_REFINED_ROOT / "miniature_toy_poodle"         # images for miniature + toy poodle
P_STAFF_DIR   = P_REFINED_ROOT / "staff_vs_american_staff"      # staffordshire bullterrier + american staffordshire terrier
P_GUARD_DIR   = P_REFINED_ROOT / "pyrenees_kuvasz"              # great pyrenees + kuvasz

OUT_ROOT = "qualitative_outputs"


# ==========================
# 5. Run qualitative generation
# ==========================
if __name__ == "__main__":
    # --- Easy cases: GT vs prediction ---
    print("\n=== Easy cases (YOLOv11n) ===")
    visualize_folder_with_gt(
        model=yolo_model,
        model_name="yolo_easy",
        images_dir=EASY_IMAGES_DIR,
        labels_dir=EASY_LABELS_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )

    print("\n=== Easy cases (RT-DETR-L) ===")
    visualize_folder_with_gt(
        model=rtdetr_model,
        model_name="rtdetr_easy",
        images_dir=EASY_IMAGES_DIR,
        labels_dir=EASY_LABELS_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )

    # --- Occlusion and clutter: GT vs prediction ---
    print("\n=== Occlusion/clutter (YOLOv11n) ===")
    visualize_folder_with_gt(
        model=yolo_model,
        model_name="yolo_occlusion",
        images_dir=OCCL_IMAGES_DIR,
        labels_dir=OCCL_LABELS_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )

    print("\n=== Occlusion/clutter (RT-DETR-L) ===")
    visualize_folder_with_gt(
        model=rtdetr_model,
        model_name="rtdetr_occlusion",
        images_dir=OCCL_IMAGES_DIR,
        labels_dir=OCCL_LABELS_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )

    # --- Fine-grained confusion pairs: predictions only ---
    print("\n=== Fine-grained: poodle pair (YOLOv11n) ===")
    visualize_folder_predictions_only(
        model=yolo_model,
        model_name="yolo_poodles",
        images_dir=P_POODLES_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )

    print("\n=== Fine-grained: poodle pair (RT-DETR-L) ===")
    visualize_folder_predictions_only(
        model=rtdetr_model,
        model_name="rtdetr_poodles",
        images_dir=P_POODLES_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )

    print("\n=== Fine-grained: staffordshire pair (YOLOv11n) ===")
    visualize_folder_predictions_only(
        model=yolo_model,
        model_name="yolo_staffordshire",
        images_dir=P_STAFF_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )

    print("\n=== Fine-grained: staffordshire pair (RT-DETR-L) ===")
    visualize_folder_predictions_only(
        model=rtdetr_model,
        model_name="rtdetr_staffordshire",
        images_dir=P_STAFF_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )

    print("\n=== Fine-grained: guardian pair (YOLOv11n) ===")
    visualize_folder_predictions_only(
        model=yolo_model,
        model_name="yolo_guardian",
        images_dir=P_GUARD_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )

    print("\n=== Fine-grained: guardian pair (RT-DETR-L) ===")
    visualize_folder_predictions_only(
        model=rtdetr_model,
        model_name="rtdetr_guardian",
        images_dir=P_GUARD_DIR,
        out_dir=OUT_ROOT,
        class_names=CLASS_NAMES,
    )
