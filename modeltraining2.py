from ultralytics import YOLO
from datetime import datetime
import cv2
import os
import torch

def convert_to_grayscale(image_dir):
    """
    Convert all images in a directory to grayscale.

    Parameters:
    - image_dir (str): Directory containing the images to convert.
    """
    for subdir, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(img_path, gray_img)

def train_yolov8_nano(data_yaml, epochs, batch_size, img_size, save_path, patience):
    """
    Train a YOLOv8 nano model with specified hyperparameters and save the trained model.

    Parameters:
    - data_yaml (str): Path to the YAML file with dataset configuration.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - img_size (int): Image size for resizing during training.
    - save_path (str): Path to save the trained model.
    - patience (int): Number of epochs with no improvement to stop training early.
    """
    # Verify GPU availability
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        device = 0  # Use the first GPU
    else:
        print("GPU is not available. Training will use the CPU.")
        device = 'cpu'

    # Load YAML file to extract dataset paths
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Convert train, val, and test datasets to grayscale
    for key in ['train', 'val', 'test']:
        if key in data_config:
            convert_to_grayscale(data_config[key])

    # Initialize YOLOv8 Nano model
    model = YOLO('yolov8n.yaml')  # Specify the YOLOv8 nano architecture

    # Train the model with early stopping
    model.train(
        data=data_yaml,  # Path to dataset YAML file
        epochs=epochs,  # Number of epochs
        batch=batch_size,  # Batch size
        imgsz=img_size,  # Image size
        patience=patience,  # Early stopping patience
        device=device  # Specify device (GPU or CPU)
    )

    # Save the trained model
    model.save(save_path)

if __name__ == "__main__":
    # Specify the YAML file path
    data_yaml_path = "data/data.yaml"  

    # Define training hyperparameters
    num_epochs = 30  # Number of epochs
    batch_size = 10  # Batch size
    img_size = 928  # Image size 
    early_stopping_patience = 5  # Early stopping patience

    # Path to save the trained model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_model_path = f"models/model_{timestamp}.pt" 

    # Train the YOLOv8 Nano model and save it
    train_yolov8_nano(data_yaml_path, num_epochs, batch_size, img_size, save_model_path, early_stopping_patience)
