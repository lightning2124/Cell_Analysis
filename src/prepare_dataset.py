"""
Dataset Preparation Script

This script separates images and labels for YOLO format datasets into train and validation sets.

Features:
- Splits data into 80% training and 20% validation sets
- Handles both image and label files
- Supports .jpg, .jpeg, and .png image formats (case-insensitive)
- Uses argparse for command-line arguments
- Provides a summary of the data separation process
- Includes error handling for missing directories

Usage:
python prepare_dataset.py <input_directory> <output_directory>

Note: The script copies image and label files from the 'Images' and 'Labels' folders 
in the input directory to a new directory structure in the specified output directory:
- output_directory/Images/train
- output_directory/Images/val
- output_directory/Labels/train
- output_directory/Labels/val

Ensure that the input directory contains both 'Images' and 'Labels' folders 
with appropriate image files (.jpg, .jpeg, .png) and corresponding label files (.txt).
"""

## Data Separation for YOLO Format

# Import necessary libraries
import os
import shutil
import random
import argparse
import sys

# Set random seed for reproducibility
random.seed(42)

# Setup argument parser
parser = argparse.ArgumentParser(description='Prepare YOLO dataset by separating images and labels into train and validation sets.')
parser.add_argument('input_dir', type=str, help='Input directory containing Images and Labels folders')
parser.add_argument('output_dir', type=str, help='Output directory for train and validation sets')
args = parser.parse_args()

## Setup Directories

# Define source directories
src_images = os.path.join(args.input_dir, 'Images')
src_labels = os.path.join(args.input_dir, 'Labels')

# Check if input directory exists
if not os.path.exists(args.input_dir):
    print(f"Error: Input directory '{args.input_dir}' does not exist.")
    sys.exit(1)

# Check if Images and Labels folders exist
if not os.path.exists(src_images) or not os.path.exists(src_labels):
    print(f"Error: Input directory must contain both 'Images' and 'Labels' folders.")
    print(f"Images folder exists: {os.path.exists(src_images)}")
    print(f"Labels folder exists: {os.path.exists(src_labels)}")
    sys.exit(1)

# Define destination directories
dest_train_images = os.path.join(args.output_dir, 'Images', 'train')
dest_val_images = os.path.join(args.output_dir, 'Images', 'val')
dest_train_labels = os.path.join(args.output_dir, 'Labels', 'train')
dest_val_labels = os.path.join(args.output_dir, 'Labels', 'val')

# Create destination directories if they don't exist
for dir in [dest_train_images, dest_val_images, dest_train_labels, dest_val_labels]:
    os.makedirs(dir, exist_ok=True)

## Prepare File Lists

# Get list of image files (case-insensitive)
image_files = [f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Check if any image files were found
if not image_files:
    print(f"Error: No image files found in '{src_images}'.")
    print("Make sure the 'Images' folder contains .jpg, .jpeg, or .png files (case-insensitive).")
    sys.exit(1)

# Shuffle the list of image files
random.shuffle(image_files)

# Calculate split index (80% for training)
split_index = int(0.8 * len(image_files))

# Split the files into train and validation sets
train_files = image_files[:split_index]
val_files = image_files[split_index:]

## Copy Files

def copy_files(file_list, src_img, src_lbl, dest_img, dest_lbl):
    for file in file_list:
        # Copy image file
        shutil.copy(os.path.join(src_img, file), os.path.join(dest_img, file))
        
        # Copy corresponding label file if it exists
        label_file = os.path.splitext(file)[0] + '.txt'
        if os.path.exists(os.path.join(src_lbl, label_file)):
            shutil.copy(os.path.join(src_lbl, label_file), os.path.join(dest_lbl, label_file))
        else:
            print(f"Warning: Label file not found for {file}")

# Copy training files
copy_files(train_files, src_images, src_labels, dest_train_images, dest_train_labels)

# Copy validation files
copy_files(val_files, src_images, src_labels, dest_val_images, dest_val_labels)

## Print Summary

print(f"\nData separation completed successfully!")
print(f"Total images: {len(image_files)}")
print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")
print(f"Train and validation sets are saved in: {args.output_dir}")