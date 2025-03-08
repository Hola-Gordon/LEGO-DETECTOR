import os
import yaml
import argparse
from ultralytics import YOLO


def train_model(data_yaml, model_type='yolov5n', epochs=10, img_size=640, batch_size=16, 
                project='runs/train', name='lego_detector', device=''):
    """Train a YOLOv5 model on the LEGO dataset"""
    # Validate YAML exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")
    
    print(f"Training with settings:")
    print(f"- Data: {data_yaml}")
    print(f"- Model: {model_type}")
    print(f"- Epochs: {epochs}")
    print(f"- Image size: {img_size}")
    print(f"- Batch size: {batch_size}")
    print(f"- Device: {device if device else 'Default'}")
    
    # Initialize model (download pretrained weights if needed)
    model = YOLO(f'{model_type}.pt')
    
    # Train the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project,
        name=name,
        device=device
    )
    
    print(f"Training complete. Model saved to {os.path.join(project, name)}")
    return os.path.join(project, name, 'weights', 'best.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LEGO detector using YOLOv5")
    parser.add_argument("--data", required=True, help="Path to data.yaml file")
    parser.add_argument("--model", default="yolov5n", choices=["yolov5n", "yolov5s", "yolov5m", "yolov5l"], 
                       help="Model type (yolov5n, yolov5s, etc.)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--img-size", type=int, default=640, help="Training image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="Device to use (e.g., '0' for GPU 0, '' for all available)")
    
    args = parser.parse_args()
    train_model(args.data, args.model, args.epochs, args.img_size, args.batch_size, device=args.device)