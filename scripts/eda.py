"""
EDA Script for our Multiclass Dog bred detection classifications
---------------------------------------------------------------
Performs dataset exploration:
- Counts breeds and image totals
- Plots breed distribution with a horizontal bar chart
- Optionally displays a few sample images per top breeds
- Dataset is from Stamford and dully referenced
"""

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Paths
BASE_DIR = os.getcwd()
DATA_DIR = r"data/images"
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Count breeds and image totals ---
breed_counts = []
for breed_folder in os.listdir(DATA_DIR):
    breed_path = os.path.join(DATA_DIR, breed_folder)
    if os.path.isdir(breed_path):
        num_images = len([f for f in os.listdir(breed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        breed_counts.append((breed_folder, num_images))

df_counts = pd.DataFrame(breed_counts, columns=["Breed", "Image_Count"]).sort_values(by="Image_Count", ascending=False)
total_images = df_counts["Image_Count"].sum()
print(f"\nFound {len(df_counts)} breeds with a total of {total_images} images.\n")
print(df_counts.head(10))

# --- Plot Breed Distribution ---
plt.figure(figsize=(14, 10))
plt.barh(df_counts["Breed"], df_counts["Image_Count"], color='skyblue')
plt.xlabel("Number of Images")
plt.ylabel("Dog Breeds")
plt.title("Stanford Dogs Dataset: Image Count per Breed")
plt.gca().invert_yaxis()  # largest bar on top
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_breed_distribution_h.png"), dpi=300)
plt.close()
print(f"Saved horizontal breed distribution plot to {os.path.join(FIGURES_DIR, 'eda_breed_distribution_h.png')}")

# --- Optional: Show 1 sample image per top 5 breeds ---
TOP_N = 5
top_breeds = df_counts.head(TOP_N)["Breed"].tolist()

plt.figure(figsize=(15, 8))
for idx, breed in enumerate(top_breeds):
    breed_path = os.path.join(DATA_DIR, breed)
    images = [f for f in os.listdir(breed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if images:
        sample_img_path = os.path.join(breed_path, random.choice(images))
        with Image.open(sample_img_path) as img:
            plt.subplot(1, TOP_N, idx+1)
            plt.imshow(img)
            plt.title(breed)
            plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_top_breeds_samples.png"), dpi=300)
plt.close()
print(f"Saved sample images for top {TOP_N} breeds to {os.path.join(FIGURES_DIR, 'eda_top_breeds_samples.png')}")

print("\n Breed distribution EDA complete. Figures saved in 'figures' folder.\n")
