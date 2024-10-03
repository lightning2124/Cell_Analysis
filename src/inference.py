"""
The main function, plot_inference, performs the following tasks:
1. Reads an input image from a given path.
2. Loads a YOLO model and performs inference on the image, or uses an existing result.
3. Draws bounding boxes around detected objects on the image.

Functions:
- plot_inference: Performs object detection and draws bounding boxes on the input image.

Note:
The confidence threshold for object detection can be adjusted using the conf_thres parameter.
"""

import cv2
import numpy as np
from ultralytics import YOLO

def plot_inference(img_path, model_path, conf_thres=0.5, existing_result=None):
    # Read the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if existing_result is None:
        # Load the model and perform inference if no existing result
        model = YOLO(model_path)
        results = model(img, conf=conf_thres)
    else:
        # Use the existing result
        results = [existing_result]

    # Draw bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color (BGR format)

    return img