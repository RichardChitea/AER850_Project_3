import os
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 Nano model
model = YOLO('runs/detect/train15/weights/best.pt')  # Replace with your model path

# Path to the evaluation images
image_directory = 'data/evaluation/'

# Directory to save the annotated images
output_directory = 'data/annotated_images/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get list of image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('png', 'jpg', 'jpeg'))]

# Get the class names from the model
class_names = model.names  # This is typically a list of class names that YOLO uses

# Function to generate a unique color for each class
def get_class_color(class_id):
    np.random.seed(class_id)  # Ensure the same class always gets the same color
    color = np.random.randint(0, 255, size=3).tolist()  # Random RGB color
    return tuple(color)

# Loop over all images and predict
for img_file in image_files:
    img_path = os.path.join(image_directory, img_file)
    
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    
    # Use model.predict() for inference (predicting bounding boxes, labels, and scores)
    results = model.predict(img)  # This performs inference
    
    # Manually annotate the image with predictions
    for result in results[0].boxes:
        # Get label, confidence, and bounding box details
        class_id = int(result.cls.item())  # Convert the label tensor to a native Python type
        confidence = result.conf.item()  # Convert the confidence tensor to a native Python type
        x1, y1, x2, y2 = result.xyxy[0].tolist()  # Convert coordinates to a list of floats
        
        # Get the class name and a unique color for the class
        class_name = class_names[class_id]
        color = get_class_color(class_id)
        
        # Define the font size for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Adjust this value for smaller/larger text
        thickness = 1
        
        # Prepare the text string
        text = f"{class_name} {confidence:.2f}"  # Display class name and confidence
        
        # Draw the bounding box with the unique color
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Put text above the bounding box with the same color as the box
        cv2.putText(img, text, (int(x1), int(y1) - 10), font, font_scale, color, thickness)
    
    # Save the annotated image to the output directory
    output_path = os.path.join(output_directory, img_file)
    cv2.imwrite(output_path, img)  # Save the annotated image
    
    print(f"Saved annotated image: {output_path}")
