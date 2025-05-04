# Model Acquisition Guide

This document outlines the process for acquiring the Tiny YOLO model architecture that will be used for chair detection.

## Source Architecture

We'll use the Tinier YOLO architecture from Xilinx's QNN-MO-PYNQ repository:
```
https://github.com/Xilinx/QNN-MO-PYNQ/blob/97bc264ce21db2883aaadafda8ad4c2f9ac31296/qnn/params/tinier-yolo-bwn-3bit-relu-nomaxpool.cfg
```

This model is a binarized weight network (BWN) variant of Tiny YOLO with the following specifications:
- 3-bit activations
- ReLU activation functions
- No max pooling layers

## Acquisition Steps

1. Clone the QNN-MO-PYNQ repository:
   ```bash
   git clone https://github.com/Xilinx/QNN-MO-PYNQ.git
   cd QNN-MO-PYNQ
   git checkout 97bc264ce21db2883aaadafda8ad4c2f9ac31296
   ```

2. Extract the model configuration:
   ```bash
   cp qnn/params/tinier-yolo-bwn-3bit-relu-nomaxpool.cfg /workspaces/FPGADetector/data/models/
   ```

3. Adapt the configuration file for chair detection:
   The original configuration is designed for multiple classes. We'll modify it to focus on a single class (chairs) by changing the number of classes and adjusting the output layer accordingly.

## Model Configuration

The Tinier YOLO model has a reduced network architecture compared to the standard Tiny YOLO, with:
- Fewer convolutional layers
- Reduced filter sizes
- Binary weights to minimize computational requirements

These optimizations make it suitable for deployment on resource-constrained FPGA platforms like the PYNQ Z2.

## Next Steps

After acquiring the model configuration, proceed to [Dataset Preparation and Training](training.md) to train the model on a custom dataset of chair images.