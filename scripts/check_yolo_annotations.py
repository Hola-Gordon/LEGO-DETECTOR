import os
import random
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np


def check_yolo_annotations(labels_dir, img_dir, num_samples=50, output_dir=None):
    """
    Visualize random samples with YOLO format annotations to verify quality.
    
    Args:
        labels_dir: Directory containing YOLO format label files (.txt)
        img_dir: Directory containing image files
        num_samples: Number of random samples to examine
        output_dir: Directory to save visualization results
    """
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    if not label_files:
        print(f"No label files found in {labels_dir}")
        return
    
    print(f"Found {len(label_files)} label files")
    
    # Randomly select samples
    samples = random.sample(label_files, min(num_samples, len(label_files)))
    
    # Create figure with subplots (5x5 grid, might need multiple pages)
    rows, cols = 5, 5
    samples_per_page = rows * cols
    
    for page in range((len(samples) + samples_per_page - 1) // samples_per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        # Get samples for this page
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(samples))
        page_samples = samples[start_idx:end_idx]
        
        print(f"Processing page {page+1} ({len(page_samples)} samples)")
        
        # Process each sample
        for i, label_file in enumerate(page_samples):
            ax = axes[i]
            base_name = os.path.splitext(label_file)[0]
            
            # Find image file
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_img = os.path.join(img_dir, base_name + ext)
                if os.path.exists(potential_img):
                    img_file = potential_img
                    break
            
            if img_file is None:
                ax.text(0.5, 0.5, f"Image not found\n{base_name}", 
                       ha='center', va='center')
                continue
            
            # Load image
            img = cv2.imread(img_file)
            if img is None:
                ax.text(0.5, 0.5, f"Failed to load\n{base_name}", 
                       ha='center', va='center')
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            
            # Display image
            ax.imshow(img)
            
            # Read YOLO format labels
            try:
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    lines = f.readlines()
                
                # Draw bounding boxes
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class, x_center, y_center, w, h
                        # Convert YOLO format to pixel coordinates
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        w = float(parts[3]) * width
                        h = float(parts[4]) * height
                        
                        # Calculate box coordinates
                        xmin = int(x_center - w/2)
                        ymin = int(y_center - h/2)
                        box_w = int(w)
                        box_h = int(h)
                        
                        # Draw rectangle
                        rect = plt.Rectangle((xmin, ymin), box_w, box_h, 
                                           fill=False, edgecolor='red', linewidth=2)
                        ax.add_patch(rect)
                
                ax.set_title(f"{base_name}", fontsize=8)
                
            except Exception as e:
                ax.imshow(img)
                ax.text(0.5, 0.5, f"Error: {str(e)[:20]}", 
                       ha='center', va='center', color='red',
                       transform=ax.transAxes)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        # Save or display
        if output_dir:
            out_file = os.path.join(output_dir, f"annotations_page{page+1}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"Saved page {page+1} to {out_file}")
        else:
            plt.show()
    
    print(f"Checked {len(samples)} random samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check random YOLO annotations in a LEGO dataset")
    parser.add_argument("--labels-dir", required=True, help="Directory containing YOLO format label files")
    parser.add_argument("--img-dir", required=True, help="Directory containing image files")
    parser.add_argument("--samples", type=int, default=50, help="Number of random samples to check")
    parser.add_argument("--output-dir", help="Directory to save visualization results")
    
    args = parser.parse_args()
    
    check_yolo_annotations(args.labels_dir, args.img_dir, args.samples, args.output_dir)