import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir,"motherboard_image.jpeg")
output_path = os.path.join(script_dir,"maskingoutput")

def create_object_mask(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if image is None:
        print("Error: Could not load image.")
        return
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to isolate the motherboard
    _, binary_mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)  # Adjust threshold as needed

    # Find contours of the motherboard
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No contours found.")
        return

    # Find the largest contour (assumes the motherboard is the largest object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Refine the mask (optional, for cleaner edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Save the resulting image
    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
    output_image_path = os.path.join(output_path, "motherboard_with_black_background.jpg")
    cv2.imwrite(output_image_path, result)
    
    print("Masking complete. Outputs saved to:", output_path)

# Provide the input image and output directory
input_image_path = os.path.join(script_dir,"motherboard_image.jpeg")
output_directory = os.path.join(script_dir, "maskingoutput\\")  

# Run the function
create_object_mask(input_image_path, output_directory)
