"""
Image Processing Script for Line Extraction

This script processes images to extract horizontal and vertical lines using computer vision techniques.
It utilizes OpenCV for image manipulation and various algorithms for line detection and analysis.

Features:
- Reads images from a specified input directory
- Applies preprocessing steps including gamma correction, blurring, and thresholding
- Extracts horizontal and vertical lines using morphological operations and Hough Transform
- Outputs processed images with detected lines overlaid

Usage:
python process_images.py <input_directory> <output_directory>

Parameters:
- input_directory: Directory containing images to be processed.
- output_directory: Directory where processed images will be saved.

Dependencies:
- OpenCV
- NumPy
- Matplotlib
- SciPy
- scikit-learn
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import matplotlib.image as mpimg


import os
import sys
import argparse
import cv2 as cv2


from IPython.display import clear_output
def plot_img(name, img):
    # Plot the grayscale image
    plt.figure(figsize=(5, 5))  # Adjust figure size as needed
    plt.imshow(img, cmap='gray')
    plt.title(name)
    plt.show()

def process_distribution(coords, section):
    if not coords:
        return []  # Return an empty list if there are no coordinates

    groups = []
    current_group = []
    for i, y in enumerate(sorted(coords)):  # Sort the y-coordinates
        if current_group and y - current_group[-1] > 75:  # Gap larger than 75 pixels
            groups.append(current_group)
            current_group = [y]
        else:
            current_group.append(y)
    if current_group:
        groups.append(current_group)

    # Filter groups based on size (optional)
    groups = [group for group in groups if len(group) > 5]

    if not groups:
        return []  # Return an empty list if there are no valid groups

    # Plot the histogram
    unique_ys, counts = np.unique(coords, return_counts=True)
  
    if section == "top" and groups:
        coords = groups[-1]
    elif section == "bottom" and groups:
        coords = groups[0]
    else:
        coords = [y for group in groups for y in group]  # Use all coordinates if not top or bottom

    # Plot the histogram
    unique_ys, counts = np.unique(coords, return_counts=True)

    # Find peaks in the histogram
    peaks, _ = find_peaks(counts, prominence=1) 

    # Calculate medians directly using KDE
    line_ys = []
    if len(peaks) > 0:
        kde = gaussian_kde(unique_ys, bw_method=0.5)  # Adjust bandwidth as needed
        x_range = np.linspace(unique_ys.min(), unique_ys.max(), 200)
        kde_values = kde(x_range)
        peaks, _ = find_peaks(kde_values)
        line_ys = x_range[peaks].tolist()  # Convert to list here

        # Handle potential unimodal clusters
        for i in range(len(peaks) - 1):
            # If the difference between two peaks is less than a threshold
            if x_range[peaks[i + 1]] - x_range[peaks[i]] < 10:  # Adjust threshold
                # Check if there are data points between these peaks
                cluster_ys = [y for y in unique_ys if x_range[peaks[i]] <= y < x_range[peaks[i + 1]]]
                if cluster_ys:
                    # Calculate the median of the potential unimodal cluster
                    median_y = np.median(cluster_ys)
                    line_ys.append(median_y)

    # Refine line positions using K-Means clustering
    if len(line_ys) > 1:
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(np.array(line_ys).reshape(-1, 1))
        cluster_labels = kmeans.labels_

        refined_line_ys = []
        for label in range(2):
            cluster_ys = np.array(line_ys)[cluster_labels == label]
            refined_line_ys.append(np.median(cluster_ys))
        line_ys = refined_line_ys

    if len(coords) < 2:
        return []  # Return an empty list if there are not enough coordinates for clustering

    kmeans = KMeans(n_clusters=min(2, len(coords)), random_state=0)
    coords = np.array(coords)  # Convert to a NumPy array
    kmeans.fit(coords.reshape(-1, 1))  # Correct reshape for k-means

    line_ys = []
    for label in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans.labels_ == label)[0]
        cluster_data = coords[cluster_indices]
        cluster_median = np.median(cluster_data)
        line_ys.append(round(cluster_median))

    return line_ys

def extract_horizontals(horizontal_line_ys, dimension):
    top, middle, bottom = [], [], []

    if horizontal_line_ys is not None:
        for line in horizontal_line_ys:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 5:  # Check for horizontal lines
                y = (y1 + y2) // 2
                if 0 <= y < dimension/3:
                    top.append(y)
                elif dimension/3 <= y < 2 * dimension /3:
                    middle.append(y)
                else:
                    bottom.append(y)
    
    grid_lines = []
    for section, name in zip([top, middle, bottom], ["top", "middle", "bottom"]):
        processed = process_distribution(section, name)
        if processed:
            grid_lines.append(processed)

    if not grid_lines:
        return np.array([[0], [0]])  # Return a default array if no lines were detected

    return np.array(grid_lines)

def extract_verticals(vertical_line_xs, dimension):
    left, middle, right = [], [], []

    if vertical_line_xs is not None:
        for line in vertical_line_xs:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 5:  # Check for vertical lines
                x = (x1 + x2) // 2
                if 0 <= x < dimension/3:
                    left.append(x)
                elif dimension/3 <= x < 2 * dimension /3:
                    middle.append(x)
                else:
                    right.append(x)
    
    grid_lines = []
    for section, name in zip([left, middle, right], ["top", "middle", "bottom"]):
        processed = process_distribution(section, name)
        if processed:
            grid_lines.append(processed)

    if not grid_lines:
        return np.array([[0], [0]])  # Return a default array if no lines were detected

    return np.array(grid_lines)
    

def extractLines(image_path, output_dir):
    # Load the image
    src = cv2.imread(image_path)
    img_height, img_width = src.shape[:2] 

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + image_path)
        return -1

    # Convert to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    gamma = 1.5  # Adjust gamma value as needed
    lookUpTable = np.empty((256), dtype='uint8')
    for i in range(256):
        lookUpTable[i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    gamma_corrected = cv2.LUT(gray, lookUpTable)

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (15, 15), 2.0)  # Increase kernel size

    # Thresholding to create a binary image
    ret3,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    bw_pre = cv2.bitwise_not(blurred)
    bw = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    cv2.THRESH_BINARY, 15, -2)  

    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    #print("V SHAPE", vertical.shape[0], vertical.shape[0]/30)
    verticalsize = rows // 30

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # Inverse vertical image
    vertical = cv2.bitwise_not(vertical)

    # Extract edges and smooth vertical image
    edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        cv2.THRESH_BINARY, 3, -2)

    # Thin the lines to 1 pixel width
    kernel = np.ones((1, 1), np.uint8)  # Use a smaller kernel for thinning
    vertical = cv2.erode(vertical, kernel, iterations=1)



    # ------------------- Horizontal Line Smoothing ------------------- #
    # Inverse horizontal image
    horizontal = cv2.bitwise_not(horizontal)

    # Extract edges and smooth horizontal image
    edges = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        cv2.THRESH_BINARY, 3, -2)

    # Thin the lines to 1 pixel width
    kernel = np.ones((1, 1), np.uint8) 
    horizontal = cv2.erode(horizontal, kernel, iterations=1)

 
    # ------------------- Overlay Lines on Original Image ------------------- #

    # Create masks for the black lines
    horizontal_mask = np.where(horizontal == 0, 255, 0).astype(np.uint8)  # Find black pixels (0)
    vertical_mask = np.where(vertical == 0, 255, 0).astype(np.uint8)  


    # Convert masks to RGB for overlay
    horizontal_mask_rgb = cv2.cvtColor(horizontal_mask, cv2.COLOR_GRAY2RGB)
    vertical_mask_rgb = cv2.cvtColor(vertical_mask, cv2.COLOR_GRAY2RGB)

    # Make lines neon green (B:0, G:255, R:0)
    horizontal_mask_rgb[:, :, 0] = 0  # Set blue channel to 0
    horizontal_mask_rgb[:, :, 2] = 0  # Set red channel to 0
    vertical_mask_rgb[:, :, 0] = 0  # Set blue channel to 0
    vertical_mask_rgb[:, :, 2] = 0  # Set red channel to 0


    # Increase line thickness by dilating the masks
    kernel = np.ones((5, 5), np.uint8) 
    horizontal_mask_rgb = cv2.dilate(horizontal_mask_rgb, kernel, iterations=1)
    vertical_mask_rgb = cv2.dilate(vertical_mask_rgb, kernel, iterations=1)

    # Overlay the lines on the original image using OpenCV
    overlay_image = cv2.addWeighted(src, 0.8, horizontal_mask_rgb, 0.2, 0) 
    overlay_image = cv2.addWeighted(overlay_image, 0.8, vertical_mask_rgb, 0.2, 0) 

    # Convert the original image to RGB format (if it's not already)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) 

    # ------------------- Extract Line Coordinates ------------------- #
    # Apply Canny edge detection
    horizontal_edges = cv2.Canny(horizontal, 50, 150) 
    vertical_edges = cv2.Canny(vertical, 50, 150) 

    for i in range(5):
        kernel = np.ones((1, 1), np.uint8) 
        horizontal_edges = cv2.erode(horizontal_edges, kernel, iterations=1)
        vertical_edges = cv2.erode(vertical_edges, kernel, iterations=1)
    

    for i in range(5):
    # Thin the lines to 1 pixel width
        # Slightly dilate the lines for more contrast
        kernel = np.ones((2, 2), np.uint8)
        horizontal_edges = cv2.dilate(horizontal_edges, kernel, iterations=1)
        vertical_edges = cv2.dilate(vertical_edges, kernel, iterations=1)

        # Apply adaptive thresholding for better contrast
        horizontal_edges = cv2.adaptiveThreshold(horizontal_edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
            cv2.THRESH_BINARY, 15, -2)
        vertical_edges = cv2.adaptiveThreshold(vertical_edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
            cv2.THRESH_BINARY, 15, -2)

        # Thin the lines to 1 pixel width
        kernel = np.ones((1,1), np.uint8) 
        horizontal_edges = cv2.erode(horizontal_edges, kernel, iterations=1)
        vertical_edges = cv2.erode(vertical_edges, kernel, iterations=1)

    
    #= Detect lines using HoughLinesP
    h_lines = cv2.HoughLinesP(horizontal_edges, 1.5, np.pi/90, 300, minLineLength=200, maxLineGap=50) 
    

    v_lines = cv2.HoughLinesP(vertical_edges, 1.5, np.pi/90, 300, minLineLength=200, maxLineGap=50) 
    

    if h_lines is None or v_lines is None or len(h_lines) == 0 or len(v_lines) == 0:
        print("No horizontal or vertical lines detected. Exiting.")
        return 
    
    print("numHLines:", len(h_lines))
    print("numVLines:", len(v_lines))

    # __________________------------------------_____________________
    # Extract line coordinates
    
    line_ys = extract_horizontals(h_lines, img_height)
    line_xs = extract_verticals(v_lines, img_width)


    # Create masks for the black lines
    horizontal_mask = np.where(edges == 255, 255, 0).astype(np.uint8)  # Find black pixels (0) 

    # Convert masks to RGB for overlay
    horizontal_mask_rgb = cv2.cvtColor(horizontal_mask, cv2.COLOR_GRAY2RGB)

    # Make lines neon green (B:0, G:255, R:0)
    horizontal_mask_rgb[:, :, 0] = 0  # Set blue channel to 0
    horizontal_mask_rgb[:, :, 2] = 0  # Set red channel to 0
    vertical_mask_rgb[:, :, 0] = 0  # Set blue channel to 0
    vertical_mask_rgb[:, :, 2] = 0  # Set red channel to 0

    

    # Overlay the lines on the original image using OpenCV
    overlay_image = cv2.addWeighted(src, 1, horizontal_mask_rgb, 1, 0)  # Change weights here
    overlay_image = cv2.addWeighted(overlay_image, 1, vertical_mask_rgb, 1, 0)  # Change weights here

    # Convert the original image to RGB format (if it's not already)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) 

    ################################################################
    line_xs = np.array(line_xs) 
    line_ys = np.array(line_ys)

    sorted_x_indices = np.argsort(line_xs, axis=1) 
    line_xs = np.take_along_axis(line_xs, sorted_x_indices, axis=1)

    sorted_y_indices = np.argsort(line_ys, axis=1) 
    line_ys = np.take_along_axis(line_ys, sorted_y_indices, axis=1)

    
    
    image = Image.open(image_path)

    # Check if line_xs and line_ys have the expected structure
    if len(line_xs) < 3 or len(line_ys) < 3:
        print("Error: Not enough line coordinates detected.")
        return  # or handle this case appropriately

    # Check if each subarray has at least 2 elements
    if any(len(subarray) < 2 for subarray in line_xs[:3]) or any(len(subarray) < 2 for subarray in line_ys[:3]):
        print("Error: Line coordinates do not have the expected structure.")
        return  # or handle this case appropriately

    # If we've passed the checks, proceed with defining A, B, C, D
    try:
        A = [line_xs[0][1], line_ys[0][1], line_xs[1][0], line_ys[1][0]]
        B = [line_xs[1][1], line_ys[0][1], line_xs[2][0], line_ys[1][0]]
        C = [line_xs[0][1], line_ys[1][1], line_xs[1][0], line_ys[2][0]]
        D = [line_xs[1][1], line_ys[1][1], line_xs[2][0], line_ys[2][0]]
    except IndexError as e:
        print(f"Error accessing line coordinates: {e}")
        print(f"line_xs shape: {np.shape(line_xs)}")
        print(f"line_ys shape: {np.shape(line_ys)}")
        return  # or handle this case appropriately
    
    # Crop the image
    A_crop = image.crop(A)
    B_crop = image.crop(B)
    C_crop = image.crop(C)
    D_crop = image.crop(D)

    thickTop = False
    thickLeft = False

    if (line_ys[2][1] - line_ys[2][0]) < (line_ys[0][1] - line_ys[0][0]):
        thickTop = True
    
    if (line_xs[2][1] - line_xs[2][0]) < (line_xs[0][1] - line_xs[0][0]):
        thickLeft = True

    
    if thickTop and thickLeft:
        print ("TL")
        A[2] = line_xs[1][1]
        A[3] = line_ys[1][1]

        B[2] =  line_xs[2][1]
        B[3] = line_ys[1][1]

        C[2] = line_xs[1][1]

        D[2] = line_xs[2][1]

    elif thickTop and (not thickLeft):
        print ("TR")
        A[3] = line_ys[1][1]

        B[0] =  line_xs[1][0]
        B[3] = line_ys[1][1]

        C[3] = line_ys[2][1]

        D[0] = line_xs[1][0]
        D[3] = line_ys[2][1]
    
    elif (not thickTop) and thickLeft:
        print("BL")
        A[1] = line_ys[0][0]
        A[2] = line_xs[1][1]

        B[1] =  line_ys[0][0]

        C[1] = line_ys[1][0]
        C[2] = line_xs[1][1]

        D[1] = line_ys[1][0]
    else:
        print("BR")
        A[0] = line_xs[0][0]

        B[0] = line_xs[1][0]

        C[0] =  line_xs[0][0]
        C[1] = line_ys[1][0]


        D[0] =  line_xs[1][0]
        D[1] = line_ys[1][0]
    
    # Crop the image
    A_crop = image.crop(A)
    B_crop = image.crop(B)
    C_crop = image.crop(C)
    D_crop = image.crop(D)


    # 1. Get the filename with extension
    filename_with_extension = os.path.basename(image_path)  # 'image_file.jpg'

    # 2. Split the extension 
    base_filename, _ = os.path.splitext(filename_with_extension)  # 'image_file'
    
    target_size = (960, 960)
    padding_color = (0, 0, 0)  # Black padding

    quadrants = {
        "A": A_crop,
        "B": B_crop,
        "C": C_crop,
        "D": D_crop
    }

    for quadrant_name, cropped_img in quadrants.items():
        # --- Padding and Resizing Logic ---
        crop_w, crop_h = cropped_img.size

        # Determine padding to create a square
        max_dimension = max(crop_w, crop_h)
        padding_right = (max_dimension - crop_w)
        padding_bottom = (max_dimension - crop_h)

        # Add padding to create a square canvas
        new_img = Image.new("RGB", (max_dimension, max_dimension), padding_color)
        new_img.paste(cropped_img, (0, 0))

        # Resize if necessary, now guaranteed to preserve aspect ratio
        if max_dimension > target_size[0]:
            new_img = new_img.resize(target_size, Image.Resampling.LANCZOS)
        # --- End of Padding and Resizing ---

        # Construct filename and save
        output_filename = f"{base_filename}_{quadrant_name}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        new_img.save(output_path)

    # --- Plot the images ---

    # Construct image paths
    image_path_a = os.path.join(output_dir, f"{base_filename}_A.jpg")
    image_path_b = os.path.join(output_dir, f"{base_filename}_B.jpg")
    image_path_c = os.path.join(output_dir, f"{base_filename}_C.jpg")
    image_path_d = os.path.join(output_dir, f"{base_filename}_D.jpg")

    # Load images
    img_a = Image.open(image_path_a)
    img_b = Image.open(image_path_b)
    img_c = Image.open(image_path_c)
    img_d = Image.open(image_path_d)

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            print(f"Processing: {input_path}")
            extractLines(input_path, output_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images using extractLines function.")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output directory for processed images")
    args = parser.parse_args()
    
    process_images(args.input_dir, args.output_dir)
