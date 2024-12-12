!pip install ultralytics
!pip install pyyaml

from ultralytics import YOLO
from datetime import datetime
import os
import torch
from google.colab import drive
drive.mount('/content/drive')

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


    # Verify GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Training will use the CPU.")

    # Initialize YOLOv8 Nano model
    model = YOLO('yolov8n.yaml')

    # Train the model with early stopping, mixed precision, and SGD optimizer
    model.train(
        data=data_yaml,  # Path to dataset YAML file
        epochs=epochs,  # Number of epochs
        batch=batch_size,  # Batch size
        imgsz=img_size,  # Image size
        patience=patience,  # Early stopping patience
        device=device,  # Specify device (GPU or CPU)
        amp=True,  # Enable mixed precision training
        optimizer='SGD',  # Use SGD optimizer
        augment=True  # Enable augmentations (boolean)
    )

    # Save the trained model
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    model.save(save_path)

    print(f"Model saved to {save_path}")

def create_save_model_path(base_dir="models", model_name="yolov8n"):
    """
    Creates a unique save path for the trained model using a timestamp.

    Args:
        base_dir (str, optional): The base directory for saving models. Defaults to "models".
        model_name (str, optional): The name of the model being trained. Defaults to "yolov8n".

    Returns:
        str: The full path for saving the model.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_model_path = os.path.join(base_dir, f"{model_name}_{timestamp}.pt")

    # Ensure the directory exists
    os.makedirs(base_dir, exist_ok=True)

    return save_model_path

# Example usage:
save_model_path = create_save_model_path()
print(f"Model will be saved to: {save_model_path}")

if __name__ == "__main__":
    # Specify the YAML file path
    from google.colab import files
    data_yaml_path = "/content/drive/My Drive/Colab Notebooks/850 P3/data/data.yaml"

    import os
    if os.path.exists(data_yaml_path):
        print("Path is correct!")
    else:
        print("Path is incorrect. Please double-check.")

    # Define training hyperparameters
    num_epochs = 100  # Number of epochs
    batch_size = 4  # Batch size
    img_size = 1280  # Image size 
    early_stopping_patience = 5  # Early stopping patience

    # Train the YOLOv8 Nano model and save it
    train_yolov8_nano(data_yaml_path, num_epochs, batch_size, img_size, save_model_path, early_stopping_patience)

import shutil
from google.colab import files

def download_folder(folder_path, zip_filename):
    """
    Downloads a folder as a zip file.

    Args:
        folder_path (str): Path to the folder to download.
        zip_filename (str): Name of the zip file to create.
    """
    shutil.make_archive(zip_filename, 'zip', folder_path)
    files.download(zip_filename + '.zip')

# Example usage:
download_folder('/content/runs/detect/train8', 'train8')  