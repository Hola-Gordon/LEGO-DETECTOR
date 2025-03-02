import yolov5

def create_model(pretrained=True, model_size='s'):
    """Create a YOLOv5 model for LEGO detection."""
    if pretrained:
        model = yolov5.load(f'yolov5{model_size}.pt')
    else:
        model = yolov5.load(f'yolov5{model_size}.yaml')
    
    model.nc = 1
    model.names = ['lego']
    # Set low confidence threshold for detection
    model.conf = 0.2
    # Set IOU threshold for NMS
    model.iou = 0.45
    return model