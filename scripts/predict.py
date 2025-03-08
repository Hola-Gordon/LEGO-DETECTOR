import os
import argparse
from ultralytics import YOLO
from PIL import Image
import glob


def predict_images(model_path, img_path, output_dir='results', conf=0.25):
    """Run prediction on images and save results"""
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Set up image paths
    if os.path.isdir(img_path):
        # Get all image files from directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(img_path, ext)))
        
        if not image_files:
            print(f"No image files found in {img_path}")
            return
    else:
        # Single image file
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image_files = [img_path]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running prediction on {len(image_files)} images with confidence threshold {conf}")
    
    # Process each image
    total_objects = 0
    for img_file in image_files:
        # Run prediction
        results = model(img_file, conf=conf)
        result = results[0]  # Get first result
        
        # Count detected objects
        num_objects = len(result.boxes)
        total_objects += num_objects
        
        # Save result image
        result_img = result.plot()
        output_name = os.path.join(output_dir, os.path.basename(img_file))
        Image.fromarray(result_img).save(output_name)
        
        print(f"- {os.path.basename(img_file)}: {num_objects} LEGO pieces detected")
    
    print(f"\nTotal objects detected: {total_objects}")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LEGO detector on images")
    parser.add_argument("--model", required=True, help="Path to model weights (.pt file)")
    parser.add_argument("--img", required=True, help="Path to image or directory of images")
    parser.add_argument("--output", default="results", help="Directory to save results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    predict_images(args.model, args.img, args.output, args.conf)