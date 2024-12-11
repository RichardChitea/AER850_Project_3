from ultralytics import YOLO
from datetime import datetime

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
    # Initialize YOLOv8 Nano model
    model = YOLO('yolov8n.yaml')  # Specify the YOLOv8 nano architecture

    # Train the model with early stopping
    model.train(
        data=data_yaml,  # Path to dataset YAML file
        epochs=epochs,  # Number of epochs
        batch=batch_size,  # Batch size
        imgsz=img_size,  # Image size
        patience=patience  # Early stopping patience
    )

    # Save the trained model
    model.save(save_path)

if __name__ == "__main__":
    # Specify the YAML file path
    data_yaml_path = "data/data.yaml"
    
    # Define training hyperparameters
    num_epochs = 30  # Number of epochs
    batch_size = 10  # Batch size
    img_size = 900  # Image size
    early_stopping_patience = 5  # Early stopping patience

    # Path to save the trained model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_model_path = f"models/model_{timestamp}.pt"

    # Train the YOLOv8 Nano model and save it
    train_yolov8_nano(data_yaml_path, num_epochs, batch_size, img_size, save_model_path, early_stopping_patience)
