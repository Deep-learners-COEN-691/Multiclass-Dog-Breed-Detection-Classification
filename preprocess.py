#%%
import numpy as np
import cv2
import os
from pathlib import Path
from lxml import etree
from sklearn.model_selection import train_test_split


#%% load the path to the image and annotation and put them in a list
image_paths = []
annotation_paths = []
root = Path("Dogbreeds")
dirs = os.listdir(root)

# %%
# write a code to remove DS_Store from dirs if it exists
if '.DS_Store' in dirs:
    dirs.remove('.DS_Store')
dirs.sort(reverse=True)    
#%% write a code that joins the dirs with the root path
img_dir = os.path.join(root, dirs[0])
anno_dir = os.path.join(root, dirs[1])

# %%
img_dirs = os.listdir(img_dir)
anno_dirs = os.listdir(anno_dir)

if ".DS_Store" in img_dirs + anno_dirs:
    img_dirs.remove(".DS_Store")
    anno_dirs.remove(".DS_Store")


# %%
img_dirss = os.path.join(img_dir, img_dirs[0])
anno_dirss = os.path.join(anno_dir, anno_dirs[0])

# %%
img_files = os.listdir(img_dirss)
anno_files = os.listdir(anno_dirss)
if ".DS_Store" in img_files + anno_files:
    img_files.remove(".DS_Store")
    anno_files.remove(".DS_Store")
# %%
IMG_dir = []
ANNO_dir = []

 

for img in img_files:
    images= os.path.join(img_dirss, img)
    IMG_dir.append(images)
for anno in anno_files:
    annotations = os.path.join(anno_dirss, anno)
    ANNO_dir.append(annotations)
    
#%%
IMGS= []
ANNO = []
for i in IMG_dir:
    temp = os.listdir(i)
    if '.DS_Store' in temp:
        temp.remove('.DS_Store')    
    for j in temp:
        IMGS.append(os.path.join(i,j))
    

for i in ANNO_dir:
    temps = os.listdir(i)
    if '.DS_Store' in temps:
        temps.remove('.DS_Store')
    for j in temps:
        ANNO.append(os.path.join(i,j))
    
# %%
import xml.etree.ElementTree as ET

#write a code that creates a dictionary, that parses the xml files and gets the width, height, name, xmin, ymin, xmax, ymax for each object in the xml file. do this for only the first xml file
root_xml_file = []
for xml_file in ANNO:
    # print(xml_file)
    tree = ET.parse(xml_file)
    treegetroot = tree.getroot()
    size = treegetroot.find("size")
    w= size.findtext('width')
    h = size.findtext('height')
    object = treegetroot.find("object")
    name = object.findtext("name")
    bndbox = object.find("bndbox")
    xmin = bndbox.findtext("xmin")
    ymin = bndbox.findtext("ymin")
    xmax = bndbox.findtext("xmax")
    ymax = bndbox.findtext("ymax")
    print(w,h,name,xmin,ymin,xmax,ymax)
    xml_dict = {
        "width":w,
        "height":h,
        "name":name,
        "xmin":xmin,
        "ymin":ymin,
        "xmax":xmax,
        "ymax":ymax
    }
    root_xml_file.append(xml_dict)

# %%
xml_coordinates = []    
for i in root_xml_file:
    width = float(i["width"])
    height = float(i["height"])
    name = i["name"]
    xmin = float(i["xmin"])
    ymin = float(i["ymin"])
    xmax = float(i["xmax"])
    ymax = float(i["ymax"])
    center_x = ((xmax - xmin) / 2) + xmin
    normarlize_center_x = round(center_x / width, 4)
    center_y = ((ymax - ymin) / 2) + ymin
    normarlize_center_y = round(center_y / height, 4)
    box_width = xmax - xmin
    normarlize_width = round(box_width / width, 4)
    box_height = ymax - ymin
    normarlize_height = round(box_height / height, 4)
    coord = [name, normarlize_center_x, normarlize_center_y, normarlize_width, normarlize_height]
    xml_coordinates.append(coord)

# %%
#create a folder
name_of_root_folder = "yolo_dataset"
os.makedirs(name_of_root_folder, exist_ok = True)
train_path = os.path.join(name_of_root_folder, "train")
os.makedirs(train_path, exist_ok = True)
train_img_path = os.path.join(train_path,"images")
os.makedirs(train_img_path, exist_ok = True)
train_label_path = os.path.join(train_path,"labels")
os.makedirs(train_label_path, exist_ok = True)

val_path = os.path.join(name_of_root_folder, "val")
os.makedirs(val_path, exist_ok = True)
val_img_path = os.path.join(val_path,"images")
os.makedirs(val_img_path, exist_ok = True)
val_label_path = os.path.join(val_path,"labels")
os.makedirs(val_label_path, exist_ok = True)

test_path = os.path.join(name_of_root_folder, "test")
os.makedirs(test_path, exist_ok = True)
test_img_path = os.path.join(test_path,"images")
os.makedirs(test_img_path, exist_ok = True)
test_label_path = os.path.join(test_path,"labels")
os.makedirs(test_label_path, exist_ok = True)

# %%
breed_list = []
for i in IMGS:
    breed = i.split("\\")[3].split("-")[1]
    breed_list.append(breed)
    
# %%
from sklearn.model_selection import train_test_split
X_train, X_rest, y_train, y_rest , breed_train, breed_rest= train_test_split(IMGS, xml_coordinates, breed_list, test_size=0.4, stratify=breed_list, random_state=42)
X_val, X_test, y_val, y_test, breed_val, breed_test = train_test_split(X_rest, y_rest, breed_rest,test_size=0.5, stratify=breed_rest, random_state=42)

# %%
# plot the distribution of breeds in the train, val, test sets
import matplotlib.pyplot as plt
import seaborn as sns
def plot_breed_distribution(breed_name, title):
    plt.figure(figsize=(12,6))
    sns.countplot(x= breed_name, order=np.unique(breed_name))
    plt.title(title)
    plt.xlabel('Breed')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()
    
plot_breed_distribution(breed_val, "Breed Distribution in Validation Set")
plot_breed_distribution(breed_test, "Breed Distribution in Test Set")
plot_breed_distribution(breed_train, "Breed Distribution in Training Set")
# %%
import shutil
from pathlib import Path
root_data_path = os.getcwd()
index = 0
for index, i in enumerate(X_train):
    scr =os.path.join(root_data_path, i)

    dest = os.path.join(root_data_path, train_img_path, f"{index}.jpg")
    shutil.copyfile(scr, dest)
index = 0  
for index, i in enumerate(X_test):
    scr = os.path.join(root_data_path, i)
    dest = os.path.join(root_data_path, test_img_path, f"{index}.jpg")
    shutil.copyfile(scr, dest)  
index = 0
for index, i in enumerate(X_val):
    scr = os.path.join(root_data_path, i)
    dest = os.path.join(root_data_path, val_img_path, f"{index}.jpg")
    shutil.copyfile(scr, dest)
    
# %%

breed_names = []
for item in img_files:
    breed_name = item.split("-")[1:] if len(item.split("-")) > 1 else item 
    breed_name = "-".join(breed_name)
    breed_names.append(breed_name)
breed_names_unique = np.unique(breed_names)
breed_names_unique = sorted([i.lower() for i in breed_names_unique])

# %%
breed_to_id = {}
for index, breed in enumerate(breed_names_unique):
    breed_to_id[breed] = index
print(breed_to_id)

# %%
for i in range(0,len(y_train)):
    coord = y_train[i]
    with open(train_label_path + f"/{i}.txt",'w') as f:
            breed = coord[0]
            print(breed)
            
            class_id = breed_to_id[coord[0].lower()]
            normarlize_center_x = coord[1]
            normarlize_center_y = coord[2]
            normarlize_width = coord[3]
            normarlize_height = coord[4]
            f.write(f"{class_id} {normarlize_center_x} {normarlize_center_y} {normarlize_width} {normarlize_height}\n")
# %%
for i in range(0,len(y_val)):
    coord = y_val[i]
    with open(val_label_path + f"/{i}.txt", 'w') as f:
        breed = coord[0]
        class_id = breed_to_id[breed.lower()]
        normarlize_center_x = coord[1]
        normarlize_center_y = coord[2]
        normarlize_width = coord[3]
        normarlize_height = coord[4]
        f.write(f"{class_id} {normarlize_center_x} {normarlize_center_y} {normarlize_width} {normarlize_height}\n" )
# %%
for i in range(0,len(y_test)):
    coord = y_test[i]
    with open(test_label_path + f"/{i}.txt", 'w') as f:
        breed = coord[0]
        class_id = breed_to_id[breed.lower()]
        normarlize_center_x = coord[1]
        normarlize_center_y = coord[2]
        normarlize_width = coord[3]
        normarlize_height = coord[4]
        f.write(f"{class_id} {normarlize_center_x} {normarlize_center_y} {normarlize_width} {normarlize_height}\n" )
# %%
#Yaml file creation
from pathlib import Path
import yaml  # pip install pyyaml

# Root of your YOLO dataset
DATASET_ROOT = Path("./yolo_dataset")

# (Optional) if you still need these elsewhere in your code
train_path = DATASET_ROOT / "train" / "images"
val_path   = DATASET_ROOT / "val"   / "images"
test_path  = DATASET_ROOT / "test"  / "images"

# Build the YOLO dataset config
yolo_config = {
    "path": str(DATASET_ROOT),        # dataset root
    "train": "train/images",
    "val":   "val/images",
    "test":  "test/images",
    "nc": len(breed_names_unique),
    # YOLO is happy with a simple list of class names:
    "names": list(breed_names_unique),
    # If you ever want index->name mapping instead, use:
    # "names": {i: name for i, name in enumerate(breed_names_unique)},
}

# Write YAML file
yolo_yaml_path = DATASET_ROOT / "yolo.yaml"
with yolo_yaml_path.open("w") as f:
    yaml.safe_dump(yolo_config, f, sort_keys=False)

print(f"Wrote YOLO config to {yolo_yaml_path}")

# %%


# %%
