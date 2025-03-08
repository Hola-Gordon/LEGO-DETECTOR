# LEGO Piece Detector

This repository contains code for training and deploying a LEGO piece detection model using YOLOv5. The model can identify and count LEGO pieces in images with high accuracy.

## Project Overview

- **Task**: Detect and count LEGO pieces in images using bounding boxes
- **Architecture**: YOLOv5 (nano and small variants)
- **Performance**: 98% mAP@0.5 on the 5,000-image dataset

## Repository Structure

```
LEGO-DETECTOR/
├── checks/                    # Directory for annotation check results
├── data/                      # Dataset directories
│   ├── lego_dataset_500/      # 500-image dataset
│   ├── lego_dataset_5000/     # 5000-image dataset
│   └── original/              # Original dataset
├── example_images/            # Example images for the Gradio app
├── results/                   # Output directory for detection visualizations
├── runs/                      # Training runs and results
│   ├── detect/                # Detection results
│   └── train/                 # Training results
│       ├── lego_detector_500/ # 500-image model
│       └── lego_detector_5000/# 5000-image model
├── scripts/                   # Python scripts
│   ├── app.py                 # Gradio web interface
│   ├── evaluate.py            # Model evaluation
│   ├── local_dataset_prep.py  # Dataset preparation
│   ├── predict.py             # Run predictions
│   ├── prepare_data.py        # Data preprocessing
│   ├── prepare_examples.py    # Prepare example images
│   ├── quick_test.py          # Quick model testing
│   └── train.py               # Model training
├── check_yolo_annotations.py  # Script for validating annotations
└── README.md                  # Project documentation
```

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/LEGO-DETECTOR.git
cd LEGO-DETECTOR
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Use the `prepare_data.py` script to prepare the dataset:

```bash
python scripts/prepare_data.py --dataset_path path/to/dataset --output_dir data/lego_dataset_500 --max_samples 500
```

## Training

Train a YOLOv5 model using the `train.py` script:

```bash
python scripts/train.py --data data/lego_dataset_500/data.yaml --model yolov5n --epochs 50
```

## Evaluation

Evaluate a trained model on the validation set:

```bash
python scripts/evaluate.py --model runs/train/lego_detector_500/weights/best.pt --data data/lego_dataset_500/data.yaml
```

## Running Predictions

Run predictions on new images:

```bash
python scripts/predict.py --model runs/train/lego_detector_500/weights/best.pt --img path/to/image.jpg
```

## Web Interface

Launch the Gradio web interface: https://huggingface.co/spaces/zanegu/LEGO_DETECTOR

```bash
# Launch from project root directory 
python scripts/app.py



## Model Performance

Two YOLOv5 models were trained and evaluated:

### YOLOv5n (500-image dataset)
- **mAP@0.5**: 0.933
- **mAP@0.5-0.95**: 0.796
- **Precision**: 0.894
- **Recall**: 0.900
- **Inference time**: ~69ms per image (on Apple M1)

### YOLOv5s (5000-image dataset)
- **mAP@0.5**: 0.980
- **mAP@0.5-0.95**: 0.938
- **Precision**: 0.978
- **Recall**: 0.957
- **Inference time**: ~231ms per image (on Apple M1)

## Acknowledgments

- The YOLOv5 implementation is based on [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- Dataset provided as part of the course assignment