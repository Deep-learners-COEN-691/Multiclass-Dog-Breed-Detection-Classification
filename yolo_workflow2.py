#%% import libs
from ultralytics import YOLO
import os, shutil, random
from pathlib import Path
import numpy as np
import cv2
from lxml import etree
from sklearn.model_selection import train_test_split

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

# %% split data into train and val set
train_img_paths, val_img_paths, train_xml_paths, val_xml_paths = train_test_split(
    IMGS_PATH_LS, XML_PATH_LS, test_size=0.2, random_state=42
)   

# %% create YOLO format labels
def create_yolo_labels(img_paths, xml_paths, img_output_dir, label_output_dir):
    label_output_dir.mkdir(parents=True, exist_ok=True)
    for img_path, xml_path in zip(img_paths, xml_paths):
        xml_objects = parse_xml_coordiantes(xml_path)
        label_filename = xml_path.stem + ".txt"
        label_filepath = label_output_dir / label_filename

        # Write YOLO formatted labels (allow empty file if no objects)
        with open(label_filepath, 'w') as label_file:
            for name, xmin, ymin, xmax, ymax, w, h in xml_objects:
                class_id = get_class_id(name)
                if class_id is None:
                    # skip unknown class names (or log if you want)
                    continue
                cx_bb, cy_bb, width_x, width_y = coord_to_yolo_bbox(xmin, ymin, xmax, ymax, w, h)
                label_file.write(f"{class_id} {cx_bb:.6f} {cy_bb:.6f} {width_x:.6f} {width_y:.6f}\n")

        shutil.copy(img_path, img_output_dir / img_path.name)
 


# %%
# Create YOLO labels and copy images for training set
create_yolo_labels(train_img_paths, train_xml_paths, train_img_dir, train_label_dir)

# Create YOLO labels and copy images for validation set
create_yolo_labels(val_img_paths, val_xml_paths, val_img_dir, val_label_dir)  
  
# %%now we can check if the files are created
print(f"Training images: {len(list(train_img_dir.glob('*.jpg')))}")
print(f"Training labels: {len(list(train_label_dir.glob('*.txt')))}")
print(f"Validation images: {len(list(val_img_dir.glob('*.jpg')))}")
print(f"Validation labels: {len(list(val_label_dir.glob('*.txt')))}")

# %% now we create a YAML file for the dataset that YOLOv8 expects
names_yaml = "\n".join([f"  - {n}" for n in breed_names])
yaml_content = (
    f"train: {str(train_img_dir.resolve())}\n"
    f"val: {str(val_img_dir.resolve())}\n"
    f"nc: {len(breed_names)}\n"
    f"names:\n{names_yaml}\n"
)

yaml_path = out_root / "dogs_dataset.yaml"
with open(yaml_path, 'w') as yaml_file:
    yaml_file.write(yaml_content)

print(f"YAML file created at: {yaml_path}")
print(yaml_content)

# load model and train once
model = YOLO("yolov8n.pt")
model.train(data=str(yaml_path), epochs=5, imgsz=640, batch=16, name="dog_breed_yolov8n")

# %%    Evaluate model performance  
results = model.val()  # Evaluate the model on the validation set
print(results)  # Print evaluation results
