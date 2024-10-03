# Cell Analysis Application

## Overview

This cell analysis application is designed to count different types of cells in microscope slides, potentially numbering in the hundreds. It aims to accelerate the research process by automating cell detection and classification.

## Table of Contents

1. [Features](#features)
4. [Usage](#usage)
5. [Modules](#modules)
6. [Notebooks](#notebooks)
7. [Workflow](#workflow)
8. [Dependencies](#dependencies)
9. [Notes](#notes)

## Features

- Automated cell detection and classification using YOLOv8
- Image preprocessing for enhanced cell visibility
- Dataset preparation for YOLO format
- Model training and evaluation
- Visualization of detected cells



## Modules

### 1. inference.py

This module provides functionality for object detection inference using YOLO models.

Key function:
- `plot_inference`: Performs object detection and draws bounding boxes on the input image.

### 2. prepare_dataset.py

This script separates images and labels for YOLO format datasets into train and validation sets.

Usage:
```bash
python prepare_images.py <input_directory> <output_directory>
```


### 3. process_image.py

This script processes images to extract horizontal and vertical lines using computer vision techniques.

Usage:
```bash
python process_images.py <input_directory> <output_directory>
```
For detailed information on the preprocessing steps, please see IMG_PROCESSING.md.


## Notebooks

### 1. training.ipynb

This notebook sets up and trains a YOLOv8 model for detecting cells in microscope images.

Key components:
- Environment setup
- Model initialization
- Data configuration
- Model training

### 2. demo.ipynb

This notebook processes microscope images of Peripheral Blood Mononuclear Cells (PBMCs) and uses a YOLOv8 model to detect and visualize the cells.

Key components:
- Image processing
- YOLO model inference
- Visualization of detected cells

## Workflow

1. **Image Processing**:
   Use `process_image.py` to preprocess your raw microscope images:


2. **Dataset Preparation**:
Use `prepare_dataset.py` to split your processed images into a YOLO dataset format:


3. **Model Training**:
Open and run `notebooks/training.ipynb` to train your YOLO model on the prepared dataset.

4. **Demo and Visualization**:
Use `notebooks/demo.ipynb` to visualize the results of your trained model on sample images.

## Dependencies

- OpenCV
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- ultralytics (YOLOv8)


## Notes

- The `annotator.pt` model used in the demo notebook is a prototype and not the final version. The final version, capable of detecting more classes of cellular structures, is not publicly available.
- Ensure that your input images are in the correct format and directory structure as expected by each script.
- Adjust parameters in the scripts and notebooks as needed for your specific use case.

