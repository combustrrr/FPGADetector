# Dataset Preparation and Training Guide

This document outlines the process for preparing a custom chair dataset and training the Tiny YOLO model on CPU.

## Dataset Preparation

### Data Collection
1. Collect at least 300-500 images of chairs with varied:
   - Types (office chairs, dining chairs, etc.)
   - Viewpoints (front, side, 45-degree angle)
   - Lighting conditions
   - Backgrounds
   - Occlusions (partially visible chairs)

2. Image requirements:
   - Format: JPG or PNG
   - Resolution: At least 416×416 pixels (or will be resized to this)
   - Clear visibility of the target object

### Dataset Structure
Organize the dataset in the YOLO format:
```
FPGADetector/data/dataset/
├── images/              # Contains all training images
│   ├── train/           # Training images
│   └── val/             # Validation images
└── labels/              # Contains annotation files
    ├── train/           # Training annotations
    └── val/             # Validation annotations
```

### Data Annotation
1. Use annotation tools like [LabelImg](https://github.com/tzutalin/labelImg) to create bounding box annotations.
2. Save annotations in YOLO format: one text file per image with each line in format:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   where coordinates are normalized to [0,1].
3. For our single-class (chair) detection, class_id will always be 0.

## Model Training

### Environment Setup
1. Set up Darknet framework for YOLO training:
   ```bash
   git clone https://github.com/AlexeyAB/darknet.git
   cd darknet
   ```

2. Modify the Makefile to enable CPU training:
   ```
   GPU=0
   CUDNN=0
   OPENCV=1
   ```

3. Compile Darknet:
   ```bash
   make
   ```

### Configuration Files
1. Create a configuration file for chair detection (chair.data):
   ```
   classes = 1
   train = /workspaces/FPGADetector/data/dataset/train.txt
   valid = /workspaces/FPGADetector/data/dataset/val.txt
   names = /workspaces/FPGADetector/data/dataset/chair.names
   backup = /workspaces/FPGADetector/data/models/
   ```

2. Create a names file (chair.names):
   ```
   chair
   ```

3. Modify the Tinier YOLO configuration file for chair detection:
   - Update the number of classes to 1
   - Adjust the number of filters in the last convolutional layer before each YOLO layer to (classes + 5) * 3 = 18

### Training Process
1. Start training:
   ```bash
   ./darknet detector train /workspaces/FPGADetector/data/dataset/chair.data /workspaces/FPGADetector/data/models/tinier-yolo-chair.cfg -clear 1
   ```

2. Training parameters:
   - Batch size: 64
   - Subdivisions: 8 (adjust based on CPU memory)
   - Learning rate: 0.001
   - Max batches: 6000 (for single class)

3. Monitor training progress:
   - Darknet outputs loss values during training
   - Lower loss indicates better model fit
   - Save intermediate weights periodically

4. After training completes, the final weights will be saved to:
   ```
   /workspaces/FPGADetector/data/models/tinier-yolo-chair_final.weights
   ```

### Evaluation
1. Evaluate the model on validation set:
   ```bash
   ./darknet detector map /workspaces/FPGADetector/data/dataset/chair.data /workspaces/FPGADetector/data/models/tinier-yolo-chair.cfg /workspaces/FPGADetector/data/models/tinier-yolo-chair_final.weights
   ```

2. Verify detection performance:
   ```bash
   ./darknet detector test /workspaces/FPGADetector/data/dataset/chair.data /workspaces/FPGADetector/data/models/tinier-yolo-chair.cfg /workspaces/FPGADetector/data/models/tinier-yolo-chair_final.weights <path_to_test_image>
   ```

## CPU Training Optimization
Since we're training on CPU, consider these optimizations:
- Reduce batch size and increase subdivisions
- Use a smaller input resolution during training (e.g., 224×224)
- Start with pre-trained weights if available
- Use early stopping if validation loss plateaus

## Next Steps
After training, proceed to [Model Quantization](quantization.md) to reduce the model size for FPGA deployment.