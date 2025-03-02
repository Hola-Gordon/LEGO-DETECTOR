import gradio as gr
import yolov5
import os
import argparse
import numpy as np
from PIL import Image

# Global model variable
model = None

def load_model(model_path):
    """Load YOLOv5 model."""
    global model
    if model is None:
        model = yolov5.load(model_path)
        print(f"Model loaded from {model_path}")
    return model

def detect_lego(image, confidence):
    """Detect LEGO pieces in an image using YOLOv5."""
    global model
    
    # Set confidence threshold
    model.conf = confidence
    
    # Make sure image is writable (create a copy)
    image_copy = np.array(image).copy()
    
    # Run inference
    results = model(image_copy)
    
    # Get count
    count = len(results.xyxy[0])
    
    # Render image with detections
    rendered_img = results.render()[0]
    
    # Convert to PIL image
    rendered_img = Image.fromarray(rendered_img)
    
    return rendered_img, f"Detected {count} LEGO pieces"

def create_demo(model_path):
    """Create Gradio demo for LEGO detection."""
    # Load model
    load_model(model_path)
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=detect_lego,
        inputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.Slider(minimum=0.1, maximum=0.9, value=0.2, step=0.05, label="Confidence Threshold")
        ],
        outputs=[
            gr.Image(type="pil", label="Detected LEGO Pieces"),
            gr.Textbox(label="Detection Result")
        ],
        title="LEGO Piece Detector",
        description="Upload an image to detect and count LEGO pieces.",
        examples=[
            # Add example images if available
            # ["examples/lego1.jpg", 0.25],
        ]
    )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LEGO detector with Gradio interface")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--share", action="store_true", help="Create a publicly shareable link")
    
    args = parser.parse_args()
    
    demo = create_demo(args.model_path)
    demo.launch(share=args.share)