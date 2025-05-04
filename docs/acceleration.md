# FPGA Acceleration Guide

This document outlines the process for implementing the quantized Tiny YOLO model as a hardware accelerator on the PYNQ Z2 platform.

## Overview

FPGA acceleration involves:
1. Converting the quantized model to a hardware description
2. Synthesizing the hardware design
3. Generating a bitstream for the PYNQ Z2 FPGA
4. Creating wrapper code to interface with the accelerator

We'll use Xilinx's FINN framework to handle the conversion of our quantized neural network to hardware.

## Hardware Implementation

### Prerequisites
- Quantized Tiny YOLO model (`tinier-yolo-chair-w1a3-calibrated.onnx`)
- Xilinx Vivado Design Suite (2020.1 or later)
- FINN toolchain
- PYNQ board support package

### Setup
1. Ensure Vivado is properly installed and licensed
2. Set up environment variables:
   ```bash
   export VIVADO_PATH=/path/to/Vivado
   export FINN_ROOT=/path/to/finn
   export PYNQ_IP=<IP address of PYNQ board>
   ```

### Model to Hardware Conversion

1. Create a hardware project using the FINN compiler:
   ```bash
   python $FINN_ROOT/src/finn/frontend/onnx_to_finn.py \
       --onnx=/workspaces/FPGADetector/data/models/tinier-yolo-chair-w1a3-calibrated.onnx \
       --output_json=/workspaces/FPGADetector/data/models/tinier-yolo-chair-hw.json
   ```

2. Configure the hardware build:
   ```bash
   python $FINN_ROOT/src/finn/util/create_accelerator.py \
       --config=/workspaces/FPGADetector/data/models/tinier-yolo-chair-hw.json \
       --output_dir=/workspaces/FPGADetector/data/models/hw_build \
       --target_fps=5 \
       --board=PYNQ-Z2
   ```

3. Apply FPGA resource constraints:
   - Total BRAM available on PYNQ Z2: 630 KB
   - Total DSP slices: 220
   - Total LUTs: 53,200
   - Adjust the parallelization factors if needed to fit within these constraints

### Bitstream Generation

1. Run the hardware build process:
   ```bash
   cd /workspaces/FPGADetector/data/models/hw_build
   make all
   ```
   This process may take several hours to complete.

2. The output will be a bitstream file (`tinier-yolo-chair.bit`) and hardware handoff file (`tinier-yolo-chair.hwh`) in the `output_dir`.

### Resource Utilization Optimization

If the design exceeds PYNQ Z2 resources:
1. Adjust the parallelization factors in the network layers
2. Reduce the operating frequency
3. Use more aggressive folding of operations
4. Consider further quantization of certain layers

## Accelerator Interface

### Data Preparation
1. Create pre-processing functions to:
   - Resize input images to the network input size (416×416)
   - Normalize pixel values
   - Convert the image to the appropriate format for the hardware accelerator

2. Create post-processing functions to:
   - Convert network outputs to bounding box coordinates
   - Apply non-maximum suppression
   - Map confidence scores to detection results

### Hardware Interface
1. Generate Python wrapper code for the accelerator:
   ```bash
   python $FINN_ROOT/src/finn/util/generate_pynq_driver.py \
       --config=/workspaces/FPGADetector/data/models/tinier-yolo-chair-hw.json \
       --output_dir=/workspaces/FPGADetector/src/acceleration/driver
   ```

2. The generated driver code provides:
   - Functions to load the bitstream
   - Methods to send data to the accelerator
   - Methods to retrieve results from the accelerator

## Integration with PYNQ

1. Copy the generated files to the PYNQ board:
   ```bash
   scp /workspaces/FPGADetector/data/models/hw_build/tinier-yolo-chair.bit xilinx@$PYNQ_IP:/home/xilinx/
   scp /workspaces/FPGADetector/data/models/hw_build/tinier-yolo-chair.hwh xilinx@$PYNQ_IP:/home/xilinx/
   scp -r /workspaces/FPGADetector/src/acceleration/driver xilinx@$PYNQ_IP:/home/xilinx/
   ```

2. Verify the accelerator can be loaded:
   ```python
   from pynq import Overlay
   overlay = Overlay("tinier-yolo-chair.bit")
   ```

## Performance Testing

1. Measure the inference performance:
   - Latency (time per image)
   - Throughput (frames per second)
   - Power consumption

2. Compare with CPU-based inference:
   - Create a simple benchmark to compare the FPGA implementation with CPU-based inference
   - Document the speedup and energy efficiency improvements

## Next Steps

After successfully implementing the hardware accelerator, proceed to [Deployment on PYNQ Z2](deployment.md) to create a complete chair detection system.