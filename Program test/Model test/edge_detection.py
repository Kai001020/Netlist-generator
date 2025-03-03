import os
import cv2
import numpy as np
from ultralytics import YOLO

# Function to detect pins by scanning along the edges
def detect_pins(image, bbox, expand_pixels=5):
    x_min, y_min, x_max, y_max = bbox

    # Expand the bounding box
    x_min = max(0, x_min - expand_pixels)
    y_min = max(0, y_min - expand_pixels)
    x_max = min(image.shape[1], x_max + expand_pixels)
    y_max = min(image.shape[0], y_max + expand_pixels)

    # Crop the component region
    component = image[y_min:y_max, x_min:x_max]

    # Convert to grayscale
    gray = cv2.cvtColor(component, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Store pin locations for each edge
    top_pins = []     # Pins on the top edge
    bottom_pins = []  # Pins on the bottom edge
    left_pins = []    # Pins on the left edge
    right_pins = []   # Pins on the right edge

    # Scan the top edge (check first 3 rows)
    for x in range(binary.shape[1]):
        for y in range(3):  # Check first 3 rows
            if binary[y, x] == 255:
                top_pins.append((x_min + x, y_min + y))
                break  # Stop after finding the first pin in this column

    # Scan the bottom edge (check last 3 rows)
    for x in range(binary.shape[1]):
        for y in range(binary.shape[0] - 3, binary.shape[0]):  # Check last 3 rows
            if binary[y, x] == 255:
                bottom_pins.append((x_min + x, y_min + y))
                break  # Stop after finding the first pin in this column

    # Scan the left edge (check first 3 columns)
    for y in range(binary.shape[0]):
        for x in range(3):  # Check first 3 columns
            if binary[y, x] == 255:
                left_pins.append((x_min + x, y_min + y))
                break  # Stop after finding the first pin in this row

    # Scan the right edge (check last 3 columns)
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1] - 3, binary.shape[1]):  # Check last 3 columns
            if binary[y, x] == 255:
                right_pins.append((x_min + x, y_min + y))
                break  # Stop after finding the first pin in this row

    return top_pins, bottom_pins, left_pins, right_pins

# Function for processing all images
def process_all_images():
    # Define the image folder path dynamically
    image_folder = os.path.join(PARENT_PATH, 'Model test/Test images')

    # Define the output folder for processed images
    output_folder = os.path.join(PARENT_PATH, 'Model test/Model test results')
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    # Load YOLO model
    pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
    train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

    # Check if there are train folders
    if not train_folders:
        raise FileNotFoundError("No 'train' folders found in the pose directory.")

    # Determine the latest train folder
    def extract_suffix(folder_name):
        if folder_name == "train":
            return 0
        else:
            return int(folder_name[5:])  # Extract the numeric suffix from folder names like 'train1', 'train2', etc.

    latest_train_folder = max(train_folders, key=extract_suffix)
    latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

    print(f"Loading model from: {latest_train_path}")
    model = YOLO(latest_train_path)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Process and save each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # Load image
        img = cv2.imread(image_path)

        # Run inference using YOLO
        results = model(image_path)[0]

        # Process each result
        for result in results:
            for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
                                             result.keypoints.xy.cpu().numpy(),
                                             result.boxes.xyxy.cpu().numpy()):
                class_idx = int(cls)
                object_name = results.names[class_idx]  # Get the class name from the model

                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Draw bounding box (without label)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Draw bounding box in blue

                # Detect pins using edge scanning
                top_pins, bottom_pins, left_pins, right_pins = detect_pins(img, (x_min, y_min, x_max, y_max), expand_pixels=7)

                # Draw detected pins (all in green)
                for pin in top_pins + bottom_pins + left_pins + right_pins:
                    cv2.circle(img, pin, radius=4, color=(0, 255, 0), thickness=-1)  # Draw pins in green

        # Save the processed image to the output folder
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, img)
        print(f"Processed image saved to: {output_image_path}")

# Main function
if __name__ == '__main__':
    # Get the current working directory and define the project root
    current_dir = os.getcwd()
    PARENT_PATH = os.path.dirname(current_dir)  # Parent path is one level up
    PROJECT_PATH = os.path.dirname(os.path.dirname(current_dir))  # Project path is two levels up

    # Process all images
    process_all_images()