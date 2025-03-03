import os
import argparse
from ultralytics import YOLO
from PIL import Image
import glob

def evaluate_model(model_path, data_yaml=None, img_dir=None, iou=0.5, conf=0.25):
    """Evaluate the model using validation data or provided images"""
    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = YOLO(model_path)
    
    # Validate on dataset if data_yaml is provided
    if data_yaml:
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")
        
        print(f"Evaluating model on dataset defined in {data_yaml}")
        results = model.val(data=data_yaml, iou=iou, conf=conf)
        
        # Extract metrics
        mAP50 = getattr(results, 'box', {}).get('map50', 0)
        mAP50_95 = getattr(results, 'box', {}).get('map', 0)
        
        print("\nEvaluation Results:")
        print(f"mAP@0.5: {mAP50:.4f}")
        print(f"mAP@0.5-0.95: {mAP50_95:.4f}")
    
    # Evaluate on individual images if img_dir is provided
    if img_dir:
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(img_dir, ext)))
        
        if not image_files:
            print(f"No image files found in {img_dir}")
            return
        
        print(f"Running inference on {len(image_files)} images from {img_dir}")
        
        # Create directory for results
        results_dir = os.path.join("results", os.path.basename(img_dir))
        os.makedirs(results_dir, exist_ok=True)
        
        # Process each image
        total_objects = 0
        for img_path in image_files:
            results = model(img_path, conf=conf, iou=iou)
            result = results[0]  # Get first result
            
            # Save result image
            result_img = result.plot()
            output_name = os.path.join(results_dir, os.path.basename(img_path))
            Image.fromarray(result_img).save(output_name)
            
            # Count objects
            num_objects = len(result.boxes)
            total_objects += num_objects
            
            print(f"- {os.path.basename(img_path)}: {num_objects} LEGO pieces detected")
        
        print(f"\nTotal objects detected: {total_objects}")
        print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LEGO detector model")
    parser.add_argument("--model", required=True, help="Path to model weights (.pt file)")
    parser.add_argument("--data", help="Path to data.yaml file for validation")
    parser.add_argument("--img-dir", help="Directory containing test images")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not args.data and not args.img_dir:
        parser.error("Either --data or --img-dir must be provided")
    
    evaluate_model(args.model, args.data, args.img_dir, args.iou, args.conf)