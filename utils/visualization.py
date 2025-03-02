import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

def visualize_prediction(image, boxes, scores, threshold=0.5, output_path=None):
    """
    Visualize object detection predictions on an image.
    
    Args:
        image: PIL Image or numpy array
        boxes: Bounding boxes (x1, y1, x2, y2)
        scores: Confidence scores
        threshold: Confidence threshold for visualization
        output_path: Path to save the visualization
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()
        # Denormalize if needed
        image = (image * 255).astype(np.uint8)
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # Filter by threshold
    mask = scores >= threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    
    # Add bounding boxes and scores
    for box, score in zip(filtered_boxes, filtered_scores):
        # Create rectangle
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1],
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add score text
        plt.text(
            box[0], box[1] - 5, 
            f'LEGO: {score:.2f}', 
            color='white', 
            fontsize=12, 
            bbox=dict(facecolor='red', alpha=0.5)
        )
    
    # Set title
    plt.title(f'Detected {len(filtered_boxes)} LEGO pieces')
    plt.axis('off')
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_batch(model, images, targets, device, output_dir, threshold=0.5, max_images=4):
    """
    Visualize model predictions on a batch of images.
    
    Args:
        model: Detection model
        images: List of images
        targets: List of target dictionaries
        device: Device to run the model on
        output_dir: Directory to save visualizations
        threshold: Confidence threshold for visualization
        max_images: Maximum number of images to visualize
    """
    model.eval()
    with torch.no_grad():
        # Move images to device
        imgs = [img.to(device) for img in images[:max_images]]
        
        # Get predictions
        preds = model(imgs)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Denormalize images
    denorm_images = []
    for img in imgs:
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        denorm_images.append(img)
    
    # Visualize each image with ground truth and predictions
    for i, (img, target, pred) in enumerate(zip(denorm_images, targets[:max_images], preds)):
        # Ground truth
        gt_boxes = target['boxes'].cpu().numpy()
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground truth plot
        axes[0].imshow(img)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        for box in gt_boxes:
            rect = patches.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1],
                linewidth=2, 
                edgecolor='green', 
                facecolor='none'
            )
            axes[0].add_patch(rect)
        
        # Prediction plot
        axes[1].imshow(img)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        # Filter by threshold
        mask = pred_scores >= threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        
        for box, score in zip(pred_boxes, pred_scores):
            rect = patches.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1],
                linewidth=2, 
                edgecolor='red', 
                facecolor='none'
            )
            axes[1].add_patch(rect)
            
            axes[1].text(
                box[0], box[1] - 5, 
                f'{score:.2f}', 
                color='white', 
                fontsize=10, 
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'visualization_{i}.png'))
        plt.close()

def create_prediction_image(image, boxes, scores, threshold=0.5):
    """
    Create an image with detection boxes drawn.
    
    Args:
        image: PIL Image or numpy array
        boxes: Bounding boxes (x1, y1, x2, y2)
        scores: Confidence scores
        threshold: Confidence threshold for visualization
        
    Returns:
        PIL Image with boxes drawn
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # Make a copy so we don't modify the original
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Convert tensors to numpy if needed
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Filter by threshold
    mask = scores >= threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    
    # Draw boxes and scores
    for box, score in zip(filtered_boxes, filtered_scores):
        # Draw rectangle
        draw.rectangle(
            [(box[0], box[1]), (box[2], box[3])],
            outline='red',
            width=3
        )
        
        # Draw score text
        draw.text(
            (box[0], box[1] - 15),
            f'LEGO: {score:.2f}',
            fill='red'
        )
    
    # Add count text
    draw.text(
        (10, 10),
        f'Count: {len(filtered_boxes)}',
        fill='red'
    )
    
    return draw_image