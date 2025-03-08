from ultralytics import YOLO
import os
import argparse
from PIL import Image
import glob
import time


def quick_test(model_path, image_path, conf=0.25, save_dir=None):
    """Quickly test a model on a single image or directory"""
    # Validate paths
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Load model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Set up output directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get image paths
    if os.path.isdir(image_path):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
        
        if not image_files:
            print(f"No image files found in {image_path}")
            return
    else:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return
        image_files = [image_path]
    
    # Process each image
    print(f"Testing model on {len(image_files)} images (confidence threshold: {conf})")
    
    for img_file in image_files:
        print(f"\nProcessing: {os.path.basename(img_file)}")
        
        # Run inference
        start_time = time.time()
        results = model(img_file, conf=conf)
        inference_time = time.time() - start_time
        
        # Get first result
        result = results[0]
        
        # Print detection info
        boxes = result.boxes
        print(f"- Detected objects: {len(boxes)}")
        
        if len(boxes) > 0:
            # Print confidence scores
            conf_scores = boxes.conf
            print(f"- Confidence scores: {', '.join([f'{score:.2f}' for score in conf_scores])}")
        
        print(f"- Inference time: {inference_time:.3f} seconds")
        
        # Save result if requested
        if save_dir:
            result_img = result.plot()
            output_path = os.path.join(save_dir, os.path.basename(img_file))
            Image.fromarray(result_img).save(output_path)
            print(f"- Result saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for LEGO detector model")
    parser.add_argument("--model", required=True, help="Path to model weights (.pt file)")
    parser.add_argument("--img", required=True, help="Path to image or directory of images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-dir", help="Directory to save results (optional)")
    
    args = parser.parse_args()
    quick_test(args.model, args.img, args.conf, args.save_dir)