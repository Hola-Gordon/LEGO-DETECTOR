import torch
import yolov5

def create_model(pretrained=True):
    """Create a YOLOv5 model for LEGO detection."""
    if pretrained:
        model = yolov5.load('yolov5s.pt')
    else:
        model = yolov5.load('yolov5s.yaml')
    
    # Set the number of classes to 1 (just LEGO)
    model.nc = 1
    model.names = ['lego']
    
    return model