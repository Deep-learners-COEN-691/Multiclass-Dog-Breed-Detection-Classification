#%% import libs
from ultralytics import YOLO
import os, shutil, random
from pathlib import Path
import numpy as np
import cv2
from lxml import etree
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
#%% initiale data dirs
dogs_root = Path("Dogbreeds")  #path to dataset
images_src = dogs_root / "images" / "Images" #path to images
ann_src    = dogs_root / "annotations" / "Annotation" #path to annotations
print(f"Images path: {images_src}")
print(f"Annotations path: {ann_src}")
assert images_src.exists() and ann_src.exists(), "Couldn't find Images/ and Annotation/ under dogs_root" #check if paths exist
# %% get class names

img_dirs = [d for d in os.listdir(images_src) if d != '.DS_Store']
# parse breed names and build multiple mappings: full folder, wnid (n020...), and breed name
breed_entries = []
for d in img_dirs:
    parts = d.split('-')
    wnid = parts[0]
    breed = parts[1] if len(parts) > 1 else d
    breed_entries.append((d, wnid, breed))
breed_entries = sorted(breed_entries, key=lambda x: x[2])  # sort by breed name
breed_names = [b for (_, _, b) in breed_entries]
print(breed_names)

# mappings for different possible name formats found in XML
full_dir_to_id = {}
wnid_to_id = {}
breedname_to_id = {}
for idx, (full, wnid, breed) in enumerate(breed_entries):
    full_dir_to_id[full] = idx
    wnid_to_id[wnid] = idx
    breedname_to_id[breed] = idx
print("Mappings sample:", list(breedname_to_id.items())[:5])

def get_class_id(xml_name):
    """Return class id for xml_name trying different formats."""
    if xml_name in breedname_to_id:
        return breedname_to_id[xml_name]
    if xml_name in wnid_to_id:
        return wnid_to_id[xml_name]
    if xml_name in full_dir_to_id:
        return full_dir_to_id[xml_name]
    # try splitting cases like "n02085620-Chihuahua"
    if '-' in xml_name:
        parts = xml_name.split('-')
        # try last part (breed)
        if parts[-1] in breedname_to_id:
            return breedname_to_id[parts[-1]]
        # try first part (wnid)
        if parts[0] in wnid_to_id:
            return wnid_to_id[parts[0]]
    return None

# %% create output directories
out_root = Path("dogs_yolo_dataset") #output root directory
train_img_dir = out_root /"Images"/"train"
val_img_dir = out_root /"Images"/"val"

train_label_dir = out_root /"labels"/"train"
val_label_dir = out_root /"labels"/"val"

train_img_dir.mkdir(parents=True, exist_ok=True)
val_img_dir.mkdir(parents=True, exist_ok=True)
train_label_dir.mkdir(parents=True, exist_ok=True)
val_label_dir.mkdir(parents=True, exist_ok=True)

# %% helper functions to convert xml to yolo bounding box
def coord_to_yolo_bbox(xmin, ymin, xmax,ymax, w, h):
    width_x = xmax - xmin
    width_y = ymax - ymin
    cx_bb = xmin + width_x/2
    cy_bb = ymin + width_y/2
    
    #normalize with w and h of image
    cx_bb =cx_bb/w
    cy_bb  = cy_bb/h
    
    
    width_x =width_x/w
    width_y = width_y/h
    
    
    return cx_bb, cy_bb, width_x, width_y

#%% test script for accesing xml file
#test path
xml_path = r'/Users/ememusoh/Desktop/workflow_yolo/Dogbreeds/annotations/Annotation/n02085620-Chihuahua/n02085620_7.xml'
tree = etree.parse(str(xml_path))
root = tree.getroot()

source = root.find('source')
database = source.findtext('database')

# %% helper function for accessing xml file to get yolo coordinates

def parse_xml_coordiantes(xml_path):
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    
    size = root.find("size")
    w = int(size.findtext("width")) 
    h = int(size.findtext("height"))
    
    # Collect all objects in this XML file. Each object is a list:
    # [name, xmin, ymin, xmax, ymax, image_width, image_height]
    xml_objects = []

    for obj in root.findall('object'):
        name = obj.findtext('name')
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        xml_objects.append([name, xmin, ymin, xmax, ymax, w, h])

    # Return the list of objects (may be empty if no objects found)
    return xml_objects
    
xml_objects =   parse_xml_coordiantes(xml_path)
print(xml_objects) 


# name, xmin, ymin, xmax, ymax, w, h = xml_objects[0]
# cx_bb, cy_bb, width_x, width_y = coord_to_yolo_bbox(xmin, ymin, xmax,ymax, w, h)

# print(cx_bb, cy_bb, width_x, width_y)

for name, xmin, ymin, xmax, ymax, w, h in xml_objects:
    cx_bb, cy_bb, width_x, width_y = coord_to_yolo_bbox(xmin, ymin, xmax,ymax, w, h)
    print(f"Object: {name}, YOLO bbox: {cx_bb}, {cy_bb}, {width_x}, {width_y}")
# %% get all annotation label from our dataset 
#ann_src
#get the  img_path, xml_path
IMGS_PATH_LS = []
XML_PATH_LS = []
annotation_dir_list = [d for d in os.listdir(ann_src) if d != '.DS_Store']
annotation_dir_list.sort(key=lambda i: i.split('-')[1] if '-' in i else i)

img_dir_list = [d for d in os.listdir(images_src) if d != '.DS_Store']
img_dir_list.sort(key=lambda i: i.split('-')[1] if '-' in i else i)

assert len(img_dir_list) == len(annotation_dir_list)


annotation_dir_list_filepath = []
image_dir_list_filepath = []
for (anno_breedname, img_breedname) in zip(annotation_dir_list,img_dir_list):
    annotation_dir_list_filepath.append(os.path.join(ann_src, anno_breedname)) #make file path to dog breed dir
    image_dir_list_filepath.append(os.path.join(images_src, img_breedname))

#get each xml path and also check for image exist in the image folder
for (anno_breedname_dir_filepath, img_breedname_dir_filepath) in zip(annotation_dir_list_filepath, image_dir_list_filepath):
    print(f'anno_breedname_dir_path:{anno_breedname_dir_filepath}')
    print(f'image_breedname_dir_path:{img_breedname_dir_filepath}')
    anno_xml_files = os.listdir(anno_breedname_dir_filepath)
    anno_xml_files.sort(key=lambda i:int(i.split("_")[1].split('.')[0]))
    imgs_src_files = os.listdir(img_breedname_dir_filepath)
    imgs_src_files.sort(key=lambda i:int(i.split("_")[1].split('.')[0]))
    
    
    #go in through each dir to add filename to path, then access the details
    for (xml_filename, img_filename) in zip(anno_xml_files, imgs_src_files):
        xml_path = Path(os.path.join(anno_breedname_dir_filepath, xml_filename))
        img_filepath = Path(os.path.join(img_breedname_dir_filepath, img_filename))
        
        # print(f'xml_path: {xml_path}')
        # print(f'img_filepath: {img_filepath}')
        
        #check if image file path exist, and annotation file path exist
        #check if image can be opened
        #check if xml can be accessed
        #if it fails, continue
        
        if img_filepath.exists() and xml_path.exists():
            #check if image can be opened and xml can be accessed
            try:
                img = cv2.imread(str(img_filepath))
                tree = etree.parse(str(xml_path))
                root = tree.getroot()
                if img is None or tree is None:
                    print(f"Warning: Unable to open image file {img_filepath}. Skipping this file.")
                    continue
            except Exception as e:
                print(f"Error opening image file {img_filepath}: {e}. Skipping this file.")
                print(f"Error opening XML file {xml_path}: {e}. Skipping this file.")
                continue
            #getting xml details
            IMGS_PATH_LS.append(img_filepath)
            XML_PATH_LS.append(xml_path)
print(f'length_of_img_path:{len(IMGS_PATH_LS)}')
print(f'length_of_xml_path:{len(XML_PATH_LS)}')



#%%perform EDA on the dataset
# Build a table of all annotated objects and compute statistics + plots.
sns.set(style="whitegrid", rc={"figure.figsize": (8, 5)})

eda_out = out_root / "eda"
eda_out.mkdir(parents=True, exist_ok=True)

# Build rows: one row per object in dataset
rows = []
for img_path, xml_path in tqdm(list(zip(IMGS_PATH_LS, XML_PATH_LS)), desc="Parsing XMLs"):
    try:
        objs = parse_xml_coordiantes(xml_path)
    except Exception as e:
        print(f"Failed to parse {xml_path}: {e}")
        continue
    # If no objects, still register the image (optional)
    if len(objs) == 0:
        rows.append({
            "image_path": str(img_path),
            "xml_path": str(xml_path),
            "class_name": None,
            "class_id": None,
            "xmin": None, "ymin": None, "xmax": None, "ymax": None,
            "img_w": None, "img_h": None,
            "bbox_w": None, "bbox_h": None,
            "bbox_area": None, "bbox_area_norm": None, "bbox_ar": None
        })
        continue

    for name, xmin, ymin, xmax, ymax, w, h in objs:
        cid = get_class_id(name)
        bw = xmax - xmin
        bh = ymax - ymin
        area = bw * bh
        norm_area = area / (w * h) if (w * h) > 0 else None
        ar = (bw / bh) if bh > 0 else None
        rows.append({
            "image_path": str(img_path),
            "xml_path": str(xml_path),
            "class_name": name,
            "class_id": cid,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "img_w": w, "img_h": h,
            "bbox_w": bw, "bbox_h": bh,
            "bbox_area": area, "bbox_area_norm": norm_area, "bbox_ar": ar
        })

df = pd.DataFrame(rows)
# Basic dataset counts
num_images = len(df["image_path"].unique())
num_annotations = df["class_name"].notna().sum()
num_classes = len(breed_names)
print(f"Images: {num_images}, Annotated objects: {num_annotations}, Classes (declared): {num_classes}")

# Class frequency (by object instances)
class_freq = df["class_name"].value_counts(dropna=True)
class_freq.to_csv(eda_out / "class_frequency.csv", index=True)
print("Top classes:\n", class_freq.head(10))

# Objects per image distribution
obj_per_image = df.groupby("image_path")["class_name"].count()
obj_per_image.describe().to_csv(eda_out / "objects_per_image_stats.csv")
# Plot: objects per image
plt.figure()
sns.histplot(obj_per_image, bins=range(0, int(obj_per_image.max())+2), kde=False)
plt.title("Objects per image")
plt.xlabel("Number objects")
plt.ylabel("Number images")
plt.tight_layout()
plt.savefig(eda_out / "objects_per_image_hist.png")
plt.close()

# Plot: class frequency top 30
plt.figure(figsize=(10, 6))
top_n = min(30, len(class_freq))
sns.barplot(x=class_freq.values[:top_n], y=class_freq.index[:top_n], palette="viridis")
plt.title("Top classes by instance count")
plt.xlabel("Instance count")
plt.tight_layout()
plt.savefig(eda_out / "class_frequency_top.png")
plt.close()

# BBox normalized area distribution
plt.figure()
sns.histplot(df["bbox_area_norm"].dropna(), bins=50, kde=True)
plt.title("Normalized bbox area distribution")
plt.xlabel("BBox area (normalized to image)")
plt.tight_layout()
plt.savefig(eda_out / "bbox_area_norm_hist.png")
plt.close()

# BBox aspect ratio distribution
plt.figure()
sns.histplot(df["bbox_ar"].dropna(), bins=50, log_scale=(False, True))
plt.title("BBox aspect ratio (width/height) distribution")
plt.xlabel("Aspect ratio (w/h)")
plt.tight_layout()
plt.savefig(eda_out / "bbox_aspect_ratio_hist.png")
plt.close()

# Image size scatter (unique image sizes)
img_sizes = df[["image_path", "img_w", "img_h"]].drop_duplicates().dropna()
plt.figure(figsize=(6,6))
sns.scatterplot(x="img_w", y="img_h", data=img_sizes, alpha=0.6)
plt.title("Image width x height distribution")
plt.xlabel("Width")
plt.ylabel("Height")
plt.tight_layout()
plt.savefig(eda_out / "image_size_scatter.png")
plt.close()

# Images per class (unique images containing the class)
images_per_class = df.dropna(subset=["class_name"]).groupby("class_name")["image_path"].nunique()
images_per_class.sort_values(ascending=False).head(20).to_csv(eda_out / "images_per_class_top20.csv")

# Save a summary text
with open(eda_out / "summary.txt", "w") as fh:
    fh.write(f"Total unique images: {num_images}\n")
    fh.write(f"Total annotated objects: {num_annotations}\n")
    fh.write(f"Declared classes (names list length): {num_classes}\n")
    fh.write("Top 10 classes by instance count:\n")
    fh.write("\n".join([f"{i}: {c}" for i,c in enumerate(class_freq.index[:10])]) + "\n")

# Save dataframe sample and basic stats
df.describe(include="all").to_csv(eda_out / "df_describe.csv")
df.head(200).to_csv(eda_out / "annotations_sample.csv", index=False)

# Create sample visualizations: for top classes, draw boxes on one image per class
def draw_boxes_on_image(image_path, ann_rows, out_path, label_col="class_name"):
    img = cv2.imread(image_path)
    if img is None:
        return
    for _, r in ann_rows.iterrows():
        if pd.isna(r["xmin"]):
            continue
        x1, y1, x2, y2 = int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = str(r[label_col])
        cv2.putText(img, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(out_path), img)

top_classes = class_freq.index[:6]
for cls in top_classes:
    subset = df[df["class_name"] == cls]
    if subset.empty:
        continue
    # pick a sample image for this class
    sample_row = subset.sample(1).iloc[0]
    img_path = sample_row["image_path"]
    ann_for_image = df[df["image_path"] == img_path]
    out_img_path = eda_out / f"sample_{cls.replace('/','_')}.jpg"
    draw_boxes_on_image(img_path, ann_for_image, out_img_path)

print(f"EDA outputs saved to {eda_out}")

#%%