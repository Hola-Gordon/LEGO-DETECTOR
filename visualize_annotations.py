import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random
from pathlib import Path

def visualize_annotation(image_path, xml_path, output_path=None):
    """
    Visualize an image with its XML annotation bounding boxes.
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    width = int(root.find('./size/width').text)
    height = int(root.find('./size/height').text)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Draw all objects
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        
        # Create rectangle
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                             linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(xmin, ymin - 5, name, 
                 color='white', fontsize=8, 
                 bbox=dict(facecolor='red', alpha=0.5))
    
    plt.title(f"Annotated Image: {os.path.basename(image_path)}")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_random_samples(dataset_dir, num_samples=5, output_dir=None):
    """
    Visualize random annotated samples from the dataset.
    """
    # Get paths
    image_dir = os.path.join(dataset_dir, 'images')
    anno_dir = os.path.join(dataset_dir, 'annotations')
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Random sample
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_file in samples:
        base_name = os.path.splitext(img_file)[0]
        xml_file = f"{base_name}.xml"
        
        img_path = os.path.join(image_dir, img_file)
        xml_path = os.path.join(anno_dir, xml_file)
        
        if os.path.exists(xml_path):
            if output_dir:
                out_path = os.path.join(output_dir, f"{base_name}_annotated.png")
                visualize_annotation(img_path, xml_path, out_path)
                print(f"Saved visualization to {out_path}")
            else:
                visualize_annotation(img_path, xml_path)
        else:
            print(f"Warning: No annotation found for {img_file}")

# Example usage
if __name__ == "__main__":
    # Change this to your dataset path
    dataset_dir = "data/lego-dataset"
    output_dir = "annotation_check"
    
    visualize_random_samples(dataset_dir, num_samples=5, output_dir=output_dir)