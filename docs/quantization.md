# Model Quantization Guide

This document outlines the process for quantizing the trained Tiny YOLO model for efficient deployment on the PYNQ Z2 platform.

## Overview

Quantization reduces the precision of the model's weights and activations, which:
- Decreases memory requirements
- Reduces computational complexity
- Enables efficient implementation on FPGAs
- Increases inference speed

For our PYNQ Z2 deployment, we'll use the Xilinx FINN framework to quantize the model to binary weights and low-bit activations.

## Quantization Process

### Prerequisites
- Trained Tiny YOLO model weights (`tinier-yolo-chair_final.weights`)
- FINN toolchain
- Python 3.6+ with PyTorch

### Setup
1. Clone the FINN repository:
   ```bash
   git clone https://github.com/Xilinx/finn.git
   cd finn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Model Conversion

1. Convert Darknet weights to ONNX format:
   ```bash
   python /workspaces/FPGADetector/src/quantization/darknet_to_onnx.py \
       --cfg=/workspaces/FPGADetector/data/models/tinier-yolo-chair.cfg \
       --weights=/workspaces/FPGADetector/data/models/tinier-yolo-chair_final.weights \
       --output=/workspaces/FPGADetector/data/models/tinier-yolo-chair.onnx
   ```

2. Import the ONNX model into FINN:
   ```bash
   python /path/to/finn/src/finn/util/onnx_to_finn.py \
       --input=/workspaces/FPGADetector/data/models/tinier-yolo-chair.onnx \
       --output=/workspaces/FPGADetector/data/models/tinier-yolo-chair-finn.onnx
   ```

### Quantization Steps

1. Apply weight quantization to binary (-1,+1):
   ```bash
   python /path/to/finn/src/finn/util/apply_weight_quantization.py \
       --input=/workspaces/FPGADetector/data/models/tinier-yolo-chair-finn.onnx \
       --output=/workspaces/FPGADetector/data/models/tinier-yolo-chair-w1.onnx \
       --weight-bits=1
   ```

2. Apply activation quantization to 3-bit:
   ```bash
   python /path/to/finn/src/finn/util/apply_activation_quantization.py \
       --input=/workspaces/FPGADetector/data/models/tinier-yolo-chair-w1.onnx \
       --output=/workspaces/FPGADetector/data/models/tinier-yolo-chair-w1a3.onnx \
       --activation-bits=3
   ```

### Calibration Dataset

For accurate quantization, we need to calibrate the quantization parameters using a subset of our dataset:

1. Create a calibration dataset (20-50 images from the validation set)
2. Run calibration to determine optimal quantization thresholds:
   ```bash
   python /workspaces/FPGADetector/src/quantization/calibrate_quantization.py \
       --model=/workspaces/FPGADetector/data/models/tinier-yolo-chair-w1a3.onnx \
       --dataset=/workspaces/FPGADetector/data/dataset/calibration \
       --output=/workspaces/FPGADetector/data/models/tinier-yolo-chair-w1a3-calibrated.onnx
   ```

### Validation

1. Verify quantized model accuracy:
   ```bash
   python /workspaces/FPGADetector/src/quantization/verify_quantized_model.py \
       --model=/workspaces/FPGADetector/data/models/tinier-yolo-chair-w1a3-calibrated.onnx \
       --dataset=/workspaces/FPGADetector/data/dataset/val \
       --output=/workspaces/FPGADetector/data/models/quantization_results.json
   ```

2. Compare accuracy with the original model:
   - If accuracy drop is < 5%, proceed with the quantized model
   - If accuracy drop is significant, consider:
      - Using more calibration data
      - Adjusting quantization parameters
      - Retraining with quantization-aware training

## Output Files

The quantization process produces the following files:
1. ONNX model with binary weights (`tinier-yolo-chair-w1.onnx`)
2. ONNX model with binary weights and 3-bit activations (`tinier-yolo-chair-w1a3.onnx`)
3. Calibrated quantized model (`tinier-yolo-chair-w1a3-calibrated.onnx`)

## Next Steps

After successfully quantizing the model, proceed to [FPGA Acceleration](acceleration.md) to generate the bitstream for the PYNQ Z2 platform.