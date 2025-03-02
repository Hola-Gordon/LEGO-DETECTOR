import torch
import numpy as np
from collections import defaultdict

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection_area = width * height
    
    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def calculate_mAP(model, data_loader, device, iou_threshold=0.5):
    """
    Calculate mean Average Precision for object detection.
    
    Args:
        model: The detection model
        data_loader: Dataset loader
        device: Device to run the model on
        iou_threshold: IoU threshold for a detection to be considered correct
        
    Returns:
        mAP value
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items() if k != 'masks'} for t in targets]
            
            # Run model
            outputs = model(images)
            
            # Store predictions and targets
            for output, target in zip(outputs, targets):
                all_predictions.append({
                    'boxes': output['boxes'].cpu(),
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu()
                })
                all_targets.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
    
    # Calculate AP for each class
    # Since we only have one class (lego), we'll just calculate AP for that class
    ap = calculate_ap_for_class(all_predictions, all_targets, class_id=1, iou_threshold=iou_threshold)
    
    return ap

def calculate_ap_for_class(predictions, targets, class_id, iou_threshold=0.5):
    """
    Calculate Average Precision for a specific class.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        class_id: Class ID to calculate AP for
        iou_threshold: IoU threshold for a detection to be considered correct
        
    Returns:
        AP value
    """
    # Collect all detections and ground truths for this class
    all_detections = []
    all_ground_truths = []
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        # Get detections for this class
        pred_boxes = pred['boxes'][pred['labels'] == class_id]
        pred_scores = pred['scores'][pred['labels'] == class_id]
        
        # Sort by confidence score
        sort_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sort_indices]
        pred_scores = pred_scores[sort_indices]
        
        all_detections.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'image_id': i
        })
        
        # Get ground truths for this class
        gt_boxes = target['boxes'][target['labels'] == class_id]
        all_ground_truths.append({
            'boxes': gt_boxes,
            'image_id': i,
            'matched': torch.zeros(len(gt_boxes), dtype=torch.bool)
        })
    
    # Flatten all detections for precision-recall calculation
    flat_detections = []
    for det in all_detections:
        for box_idx in range(len(det['boxes'])):
            flat_detections.append({
                'box': det['boxes'][box_idx],
                'score': det['scores'][box_idx],
                'image_id': det['image_id']
            })
    
    # Sort by confidence score
    flat_detections.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate precision and recall at each detection
    num_gts = sum(len(gt['boxes']) for gt in all_ground_truths)
    if num_gts == 0:
        return 0.0  # No ground truths for this class
    
    tp = np.zeros(len(flat_detections))
    fp = np.zeros(len(flat_detections))
    
    for i, det in enumerate(flat_detections):
        image_id = det['image_id']
        gt = all_ground_truths[image_id]
        
        best_iou = -float('inf')
        best_idx = -1
        
        # Find best matching ground truth
        for j, gt_box in enumerate(gt['boxes']):
            if gt['matched'][j]:
                continue
                
            iou = calculate_iou(det['box'].numpy(), gt_box.numpy())
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        
        if best_iou >= iou_threshold and best_idx >= 0:
            tp[i] = 1
            gt['matched'][best_idx] = True
        else:
            fp[i] = 1
    
    # Calculate cumulative precision and recall
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / num_gts
    
    # Calculate average precision using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap