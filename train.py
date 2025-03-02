# Updated train.py
import os
import subprocess
import argparse
from pathlib import Path

def main(args):
    # Set device
    device = "cpu" if not torch.cuda.is_available() else "0"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset YAML path
    dataset_yaml = os.path.join(args.data_dir, 'dataset.yaml')
    
    # Use subprocess to call yolov5 train with updated syntax
    cmd = [
        "yolo", "train", 
        f"data={dataset_yaml}",
        f"model=yolov5n.pt",  
        f"epochs={args.epochs}",
        f"batch={args.batch_size}",
        f"imgsz=320",        
        f"project={args.output_dir}",
        f"name=lego_detector",
        f"device={device}",
        "augment=True"        
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # Check if command was successful
    if result.returncode != 0:
        print("Error: Training failed!")
        return
    
    print("Training complete!")

if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser(description="Train LEGO detector using YOLOv5")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save models")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    main(args)