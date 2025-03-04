import os
import random
import shutil
from pathlib import Path
import argparse

def prepare_example_images(source_dir, target_dir, num_examples=10, seed=42):
    """
    Randomly select example images from the original dataset for app demonstration
    
    Args:
        source_dir: Directory containing original images
        target_dir: Directory to save example images
        num_examples: Number of example images to select
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files
    image_dir = os.path.join(source_dir, 'images')
    if not os.path.exists(image_dir):
        image_dir = source_dir  # In case 'images' subdirectory doesn't exist
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} image files in {image_dir}")
    
    # Randomly select images
    if len(image_files) <= num_examples:
        selected_images = image_files
        print(f"Selected all {len(selected_images)} available images")
    else:
        selected_images = random.sample(image_files, num_examples)
        print(f"Randomly selected {len(selected_images)} images")
    
    # Copy selected images to target directory
    for img_file in selected_images:
        src_path = os.path.join(image_dir, img_file)
        dst_path = os.path.join(target_dir, img_file)
        shutil.copy2(src_path, dst_path)
        print(f"Copied {img_file} to {target_dir}")
    
    print(f"\nPrepared {len(selected_images)} example images in {target_dir}")
    print("You can now use these examples in your Gradio app by modifying app.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare example images for Gradio app")
    parser.add_argument("--source", required=True, help="Source directory with original images")
    parser.add_argument("--target", default="examples", help="Target directory for example images")
    parser.add_argument("--num", type=int, default=10, help="Number of example images to select")
    
    args = parser.parse_args()
    prepare_example_images(args.source, args.target, args.num)