# Multiclass Dog-Breed-Classification Using Stanford Dogs Dataset — Setup Instructions

This project uses the **Stanford Dogs Dataset** for fine-grained visual categorization.

## Dataset Overview
- Total images: 20,580  
- Number of breeds: 120  
- Format: JPEG images with PASCAL VOC annotations  
- Source: [http://vision.stanford.edu/aditya86/ImageNetDogs/](http://vision.stanford.edu/aditya86/ImageNetDogs/)

##  How to Download
1. Visit the official dataset page:  
   http://vision.stanford.edu/aditya86/ImageNetDogs/
2. Download these two files:
   - `Images.tar`
   - `Annotation.tar`
3. Extract them and place them inside:
data/
images/
annotations/

## Note
The dataset is **not included** in this repository because of large file size (>1 GB).  
Instead, this file provides reproducible download instructions.

## Example Directory Structure
data/
images/
n02085620-Chihuahua/
image1.jpg
annotations/
n02085620-Chihuahua/
annotation1.xml