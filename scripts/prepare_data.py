import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from tqdm import tqdm

def check_annotation(xml_path):
    """Verify if the XML annotation file is valid."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Check if the annotation has at least one object
        objects = root.findall('object')
        if len(objects) == 0:
            return False
        
        # Get image dimensions
        size = root.find('size')
        if size is None:
            return False
            
        width = int(float(size.find('width').text))
        height = int(float(size.find('height').text))
        
        if width <= 0 or height <= 0:
            return False
        
        # Check if all objects have valid bounding boxes
        for obj in objects:
            bbox = obj.find('bndbox')
            if bbox is None:
                return False
                
            # Check if bounding box coordinates are valid
            try:
                xmin = max(0, int(float(bbox.find('xmin').text)))
                ymin = max(0, int(float(bbox.find('ymin').text)))
                xmax = min(width, int(float(bbox.find('xmax').text)))
                ymax = min(height, int(float(bbox.find('ymax').text)))
            except (ValueError, TypeError):
                return False
            
            # Ensure box has area and is properly formatted
            if xmin >= xmax or ymin >= ymax or (xmax-xmin)*(ymax-ymin) < 100:  # Minimum area check
                return False
        
        return True
    except Exception as e:
        print(f"Error checking annotation {xml_path}: {e}")
        return False

def convert_to_yolo_format(xml_path, output_path, img_width, img_height):
    """Convert VOC XML annotation to YOLO format."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            # All objects are class 0 (lego)
            class_id = 0
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format: center_x, center_y, width, height (normalized)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def split_dataset(dataset_path, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, max_samples=3000):
    """Split the dataset into training, validation and test sets in YOLO format."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.01, "Ratios must sum to 1"
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Get list of all image files
    image_dir = os.path.join(dataset_path, 'images')
    annotation_dir = os.path.join(dataset_path, 'annotations')
    
    valid_files = []
    print("Checking annotations...")
    
    for img_file in tqdm(os.listdir(image_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        base_name = os.path.splitext(img_file)[0]
        xml_file = f"{base_name}.xml"
        xml_path = os.path.join(annotation_dir, xml_file)
        
        # Check if annotation exists and is valid
        if os.path.exists(xml_path) and check_annotation(xml_path):
            # Verify image can be read
            try:
                img = cv2.imread(os.path.join(image_dir, img_file))
                if img is not None:
                    valid_files.append((base_name, img_file, xml_file, img.shape[1], img.shape[0]))  # Store img width, height
            except Exception as e:
                print(f"Error reading image {img_file}: {e}")
    
    print(f"Found {len(valid_files)} valid annotated images")
    
    # Limit the number of samples if specified
    if max_samples and len(valid_files) > max_samples:
        random.shuffle(valid_files)
        valid_files = valid_files[:max_samples]
        print(f"Using {len(valid_files)} samples out of the total valid files")
    
    # Shuffle and split
    random.shuffle(valid_files)
    n_samples = len(valid_files)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train+n_val]
    test_files = valid_files[n_train+n_val:]
    
    # Copy files to respective directories
    print("Copying files to train/val/test directories...")
    for file_set, target_name in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        print(f"Processing {target_name} set ({len(file_set)} files)...")
        for base_name, img_file, xml_file, img_width, img_height in tqdm(file_set):
            # Copy image
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(output_dir, target_name, 'images', img_file)
            shutil.copy(src_img, dst_img)
            
            # Convert and save annotation in YOLO format
            xml_path = os.path.join(annotation_dir, xml_file)
            dst_label = os.path.join(output_dir, target_name, 'labels', f"{base_name}.txt")
            convert_to_yolo_format(xml_path, dst_label, img_width, img_height)
    
    # Create dataset.yaml for YOLO
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"# YOLO dataset configuration\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"test: test/images\n\n")
        f.write(f"# Classes\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['lego']\n")
    
    print(f"Dataset split complete:\n"
          f"- Train: {len(train_files)}\n"
          f"- Validation: {len(val_files)}\n"
          f"- Test: {len(test_files)}")
    print(f"Created dataset.yaml at {yaml_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess LEGO dataset for YOLO")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to original dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save processed dataset")
    parser.add_argument("--max_samples", type=int, default=3000, help="Maximum number of samples to use")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training samples")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of validation samples")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of test samples")
    
    args = parser.parse_args()
    
    # Verify ratios sum to 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"The sum of train, val, and test ratios must be 1.0, got {total_ratio}")
    
    split_dataset(args.dataset_path, args.output_dir, 
                  args.train_ratio, args.val_ratio, args.test_ratio, 
                  args.max_samples)