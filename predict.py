# Updated predict.py
import os
import subprocess
import argparse
from pathlib import Path

def predict(model_path, image_path, output_dir=None, confidence_threshold=0.25):
    """Run inference on images to detect LEGO pieces."""
    # Create command
    cmd = [
        "yolo", "detect",
        "--weights", model_path,
        "--source", image_path,
        "--conf", str(confidence_threshold)
    ]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cmd.extend(["--project", output_dir])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LEGO detector on images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Path to save visualizations")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    predict(args.model_path, args.image_path, args.output_dir, args.threshold)