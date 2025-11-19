import os
import shutil

images_path = "data/images"

for filename in os.listdir(images_path):
    file_path = os.path.join(images_path, filename)
    if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            breed = filename.split("-")[1].split("_")[0]
        except IndexError:
            breed = "unknown"

        breed_folder = os.path.join(images_path, breed)
        os.makedirs(breed_folder, exist_ok=True)

        shutil.move(file_path, os.path.join(breed_folder, filename))

print("Images organized by breed ✅")
