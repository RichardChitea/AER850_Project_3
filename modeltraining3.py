from ultralytics import YOLO
from datetime import datetime
import os
import torch

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

    # Initialize YOLOv8 Nano model
    model = YOLO('yolov8n.yaml')  # Specify the YOLOv8 nano architecture

    # Train the model with early stopping, mixed precision, and memory-efficient optimizer
    try:
        model.train(
            data=data_yaml,  # Path to dataset YAML file
            epochs=epochs,  # Number of epochs
            batch=batch_size,  # Batch size
            imgsz=img_size,  # Image size
            patience=patience,  # Early stopping patience
            device=device,  # Specify device (GPU or CPU)
            amp=True,  # Enable mixed precision training
            optimizer='AdamW',  # Memory-efficient optimizer
            workers=12  # Number of data loading workers
        )
    except RuntimeError as e:
        print("Mixed precision training is not supported on this device. Retrying without AMP...")
        model.train(
            data=data_yaml,  # Path to dataset YAML file
            epochs=epochs,  # Number of epochs
            batch=batch_size,  # Batch size
            imgsz=img_size,  # Image size
            patience=patience,  # Early stopping patience
            device=device,  # Specify device (GPU or CPU)
            amp=False,  # Disable mixed precision training
            optimizer='AdamW',  # Memory-efficient optimizer
            workers=12  # Number of data loading workers
        )

    # Save the trained model
    model.save(save_path)

if __name__ == "__main__":
    # Specify the YAML file path
    data_yaml_path = "data/data.yaml"  # Replace with your YAML file path

    # Define training hyperparameters
    num_epochs = 100  # Number of epochs
    batch_size = 4  # Batch size
    img_size = 1280  # Image size (e.g., 640x640)
    early_stopping_patience = 5  # Early stopping patience

    # Path to save the trained model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_model_path = f"models/model_{timestamp}.pt"  # Replace with your desired save path

    # Train the YOLOv8 Nano model and save it
    train_yolov8_nano(data_yaml_path, num_epochs, batch_size, img_size, save_model_path, early_stopping_patience)
