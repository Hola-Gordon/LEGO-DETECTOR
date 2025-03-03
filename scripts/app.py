import gradio as gr
from ultralytics import YOLO
import os
import argparse
import numpy as np
from PIL import Image
import time

# Global model variable
model = None

def load_model(model_path):
    """Load YOLO model."""
    global model
    if model is None and os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Model loaded from {model_path}")
        return model
    elif not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    return model

def detect_lego(image, confidence):
    """Detect LEGO pieces in an image using YOLO."""
    global model
    
    if model is None:
        return image, "Error: Model not loaded"
    
    if image is None:
        return None, "Error: No image provided"
    
    try:
        # Make sure image is writable
        image_copy = np.array(image).copy()
        
        # Run inference with confidence threshold
        start_time = time.time()
        results = model(image_copy, conf=confidence)
        inference_time = time.time() - start_time
        
        # Get the first result (only one image was processed)
        result = results[0]
        
        # Get count of detected objects
        count = len(result.boxes)
        
        # Get the rendered image with detections
        rendered_img = result.plot()
        
        # Create result message
        result_msg = f"Detected {count} LEGO pieces (confidence threshold: {confidence:.2f})\n"
        result_msg += f"Inference time: {inference_time:.3f} seconds"
        
        # Convert to PIL image
        rendered_img = Image.fromarray(rendered_img)
        
        return rendered_img, result_msg
    
    except Exception as e:
        return image, f"Error during detection: {str(e)}"

def create_demo(model_path):
    """Create Gradio demo for LEGO detection."""
    # Load model
    if not load_model(model_path):
        print("Error: Could not load model")
        return None
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=detect_lego,
        inputs=[
            gr.Image(type="numpy", label="Input Image"),
            gr.Slider(minimum=0.01, maximum=0.95, value=0.25, step=0.05, label="Confidence Threshold")
        ],
        outputs=[
            gr.Image(type="pil", label="Detected LEGO Pieces"),
            gr.Textbox(label="Detection Result")
        ],
        title="LEGO Piece Detector",
        description="Upload an image to detect and count LEGO pieces.",
        examples=[
            # You can add example images here if available
        ]
    )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LEGO detector with Gradio interface")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--share", action="store_true", help="Create a publicly shareable link")
    
    args = parser.parse_args()
    
    demo = create_demo(args.model_path)
    if demo:
        demo.launch(share=args.share)
    else:
        print("Failed to create demo. Please check the model path.")