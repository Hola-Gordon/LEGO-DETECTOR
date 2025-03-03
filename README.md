# LEGO Piece Detector

This project implements an object detection system for LEGO pieces using YOLOv5. The system detects and counts LEGO pieces in images regardless of color or type.

## Project Structure

```
LEGO-DETECTOR/
├── data/
│   ├── lego_dataset_500/    # Small dataset for testing
│   ├── lego_dataset_3000/   # Larger dataset for full training
│   └── original/            # Original full dataset
├── models/                  # Trained model weights
├── scripts/
│   ├── local_dataset_prep.py  # Script for processing dataset locally
│   ├── prepare_data.py        # Data preparation and validation
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── predict.py             # Prediction script
│   ├── quick_test.py          # Quick model testing
│   └── app.py                 # Gradio web interface
└── requirements.txt         # Package dependencies
```

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/LEGO-DETECTOR.git
cd LEGO-DETECTOR
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Workflow

### 1. Data Preparation

First, prepare a smaller dataset to work with:

```bash
# On your local machine:
python scripts/local_dataset_prep.py --source data/original --target data/lego_dataset_500 --samples 500
```

Or if you're working directly with the full dataset on a cluster:

```bash
python scripts/prepare_data.py --dataset_path data/original --output_dir data/lego_dataset_3000 --max_samples 3000
```

### 2. Training

Train the model using the prepared dataset:

```bash
python scripts/train.py --data data/lego_dataset_500/data.yaml --model yolov5n --epochs 10 --img-size 640 --batch-size 16
```

For smaller machines or faster training, use:
```bash
python scripts/train.py --data data/lego_dataset_500/data.yaml --model yolov5n --epochs 5 --img-size 320 --batch-size 8
```

### 3. Evaluation

Evaluate the trained model:

```bash
python scripts/evaluate.py --model runs/train/lego_detector/weights/best.pt --data data/lego_dataset_500/data.yaml
```

### 4. Prediction

Run predictions on new images:

```bash
python scripts/predict.py --model runs/train/lego_detector/weights/best.pt --img test_images/ --output results
```

### 5. Quick Testing

For a quick test of the model:

```bash
python scripts/quick_test.py --model runs/train/lego_detector/weights/best.pt --img test_images/sample.jpg --save-dir results
```

### 6. Interactive Demo

Launch a Gradio web interface for interactive testing:

```bash
python scripts/app.py --model_path runs/train/lego_detector/weights/best.pt --share
```

## Notes on Model Performance

- The YOLOv5n (nano) model is the smallest and fastest but may have lower accuracy
- For better accuracy but slower inference, use YOLOv5s or YOLOv5m
- Reducing image size speeds up training but may reduce accuracy
- The confidence threshold controls detection sensitivity (lower = more detections)

## Requirements

- Python 3.8+
- CUDA-compatible GPU recommended for training
- See requirements.txt for package dependencies