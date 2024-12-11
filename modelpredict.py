import os
from ultralytics import YOLO
import cv2

# Load the YOLOv8 Nano model
model = YOLO('runs/detect/train13/weights/best.pt') 

# Path to the evaluation images
image_directory = 'data/evaluation/'

# Directory to save the annotated images
output_directory = 'data/annotated_images/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get list of image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('png', 'jpg', 'jpeg'))]

# Loop over all images and predict
for img_file in image_files:
    img_path = os.path.join(image_directory, img_file)
    
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    
    # Use model.predict() for inference (predicting bounding boxes, labels, and scores)
    results = model.predict(img)  
    
    # Annotate the image with predictions
    annotated_img = results[0].plot()
    
    # Save the annotated image to the output directory
    output_path = os.path.join(output_directory, img_file)
    cv2.imwrite(output_path, annotated_img) 

    print(f"Saved annotated image: {output_path}")
