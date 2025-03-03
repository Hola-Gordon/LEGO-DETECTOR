import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import zipfile

def create_compact_dataset(source_dir, target_dir, sample_size=3000, seed=42, 
                          img_size=640, compress=True):
    """Create a compact dataset for upload to cluster"""
    random.seed(seed)
    
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(target_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, 'labels'), exist_ok=True)
    
    # Find paired image and annotation files
    image_dir = os.path.join(source_dir, 'images')
    anno_dir = os.path.join(source_dir, 'annotations')
    
    print("Finding valid image-annotation pairs...")
    valid_pairs = []
    for img_file in tqdm(os.listdir(image_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        base_name = os.path.splitext(img_file)[0]
        xml_file = f"{base_name}.xml"
        xml_path = os.path.join(anno_dir, xml_file)
        
        if os.path.exists(xml_path):
            try:
                # Verify image can be opened
                img = cv2.imread(os.path.join(image_dir, img_file))
                if img is None:
                    continue
                
                # Verify annotation is valid
                tree = ET.parse(xml_path)
                root = tree.getroot()
                if len(root.findall('object')) > 0:
                    valid_pairs.append((img_file, xml_file))
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                pass
    
    print(f"Found {len(valid_pairs)} valid image-annotation pairs")
    
    # Sample subset
    if sample_size > len(valid_pairs):
        sample_size = len(valid_pairs)
        print(f"Warning: Requested sample size exceeds available data. Using {sample_size} samples.")
    
    samples = random.sample(valid_pairs, sample_size)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * sample_size)
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    
    # Process training samples
    print("Processing training samples...")
    for img_file, xml_file in tqdm(train_samples):
        process_sample(
            os.path.join(image_dir, img_file),
            os.path.join(anno_dir, xml_file),
            os.path.join(target_dir, 'train'),
            img_size
        )
    
    # Process validation samples
    print("Processing validation samples...")
    for img_file, xml_file in tqdm(val_samples):
        process_sample(
            os.path.join(image_dir, img_file),
            os.path.join(anno_dir, xml_file),
            os.path.join(target_dir, 'val'),
            img_size
        )
    
    # Create YAML file
    yaml_path = os.path.join(target_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(target_dir)}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['lego']\n")
    
    print(f"Dataset created with {len(train_samples)} training and {len(val_samples)} validation samples")
    
    # Create ZIP file if requested
    if compress:
        zip_path = f"{target_dir}.zip"
        print(f"Creating ZIP archive at {zip_path}...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(target_dir))
                    zipf.write(file_path, arcname)
        
        print(f"ZIP archive created: {zip_path}")

def process_sample(img_path, xml_path, target_dir, img_size):
    """Process a single sample: resize image and convert annotation to YOLO format"""
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Read and resize image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Save resized image
    img_save_path = os.path.join(target_dir, 'images', f"{base_name}.jpg")
    cv2.imwrite(img_save_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Convert annotation to YOLO format
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    with open(os.path.join(target_dir, 'labels', f"{base_name}.txt"), 'w') as f:
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (centered x, y, width, height - normalized)
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            bbox_width = (xmax - xmin) / w
            bbox_height = (ymax - ymin) / h
            
            # Class is always 0 (lego)
            f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create compact LEGO dataset for upload")
    parser.add_argument("--source", required=True, help="Source dataset directory")
    parser.add_argument("--target", required=True, help="Target directory for processed dataset")
    parser.add_argument("--samples", type=int, default=3000, help="Number of samples")
    parser.add_argument("--img-size", type=int, default=640, help="Image size (square)")
    parser.add_argument("--no-compress", action="store_false", dest="compress", 
                       help="Don't create ZIP archive")
    
    args = parser.parse_args()
    create_compact_dataset(args.source, args.target, args.samples, img_size=args.img_size, 
                          compress=args.compress)