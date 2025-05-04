# Deployment Guide for PYNQ Z2

This document outlines the process for deploying the accelerated Tiny YOLO chair detection system on the PYNQ Z2 platform and testing its performance.

## Overview

Deployment involves:
1. Setting up the PYNQ Z2 board
2. Installing necessary dependencies
3. Transferring the bitstream and model files
4. Creating a Python interface for chair detection
5. Testing the system with live or pre-recorded video

## PYNQ Z2 Setup

### Hardware Setup
1. Power the PYNQ Z2 board via micro USB or DC power supply
2. Connect to the board via Ethernet
3. Optional: Connect a USB camera for live detection

### Software Setup
1. Ensure the PYNQ Z2 is running the latest PYNQ image (v2.7 or later):
   ```bash
   ssh xilinx@<PYNQ_IP>
   sudo pynq_get_notebooks --upgrade
   ```

2. Install additional Python dependencies:
   ```bash
   sudo pip3 install opencv-python numpy matplotlib
   ```

## Deployment Steps

### 1. File Transfer
Transfer the following files to the PYNQ Z2:
```bash
# Create project directory on PYNQ
ssh xilinx@<PYNQ_IP> "mkdir -p ~/chair_detector"

# Copy bitstream and hardware handoff files
scp /workspaces/FPGADetector/data/models/hw_build/tinier-yolo-chair.bit xilinx@<PYNQ_IP>:~/chair_detector/
scp /workspaces/FPGADetector/data/models/hw_build/tinier-yolo-chair.hwh xilinx@<PYNQ_IP>:~/chair_detector/

# Copy driver files
scp -r /workspaces/FPGADetector/src/acceleration/driver/* xilinx@<PYNQ_IP>:~/chair_detector/

# Copy deployment scripts
scp -r /workspaces/FPGADetector/src/deployment/* xilinx@<PYNQ_IP>:~/chair_detector/
```

### 2. Application Structure
Create the following files in the PYNQ board's `chair_detector` directory:

#### a. Inference Wrapper (`chair_detector.py`)
```python
import cv2
import numpy as np
from pynq import Overlay
from chair_detector_driver import TinierYoloDriver

class ChairDetector:
    def __init__(self, bitstream_path="tinier-yolo-chair.bit"):
        # Load the overlay
        self.overlay = Overlay(bitstream_path)
        
        # Initialize the accelerator driver
        self.driver = TinierYoloDriver(self.overlay)
        
        # Detection parameters
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (416, 416)
    
    def preprocess(self, image):
        # Resize image to network input size
        resized = cv2.resize(image, self.input_size)
        
        # Convert to the format expected by the accelerator
        # (Format depends on specific accelerator implementation)
        processed = resized.astype(np.float32) / 255.0
        return processed
    
    def postprocess(self, network_output, original_img):
        # Convert network output to bounding boxes
        # Apply confidence threshold
        # Apply non-maximum suppression
        # Scale back to original image dimensions
        
        # Example (actual implementation will depend on network output format):
        h, w = original_img.shape[:2]
        boxes = []
        
        # Process network_output to get boxes, scores
        # ...
        
        return boxes, scores
    
    def detect(self, image):
        # Save original dimensions
        h, w = image.shape[:2]
        
        # Preprocess image
        processed_img = self.preprocess(image)
        
        # Run inference on FPGA
        network_output = self.driver.inference(processed_img)
        
        # Postprocess results
        boxes, scores = self.postprocess(network_output, image)
        
        return boxes, scores
    
    def draw_detections(self, image, boxes, scores):
        # Draw bounding boxes and labels on the image
        result_img = image.copy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add label
            label = f"Chair: {score:.2f}"
            cv2.putText(result_img, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_img
```

#### b. Live Detection Script (`live_detection.py`)
```python
import cv2
import time
import numpy as np
from chair_detector import ChairDetector

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize detector
    detector = ChairDetector()
    
    print("Press 'q' to quit")
    fps_list = []
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Start timing
            start_time = time.time()
            
            # Detect chairs
            boxes, scores = detector.detect(frame)
            
            # Draw results
            result_frame = detector.draw_detections(frame, boxes, scores)
            
            # Calculate FPS
            end_time = time.time()
            elapsed = end_time - start_time
            fps = 1 / elapsed
            fps_list.append(fps)
            avg_fps = sum(fps_list[-30:]) / min(len(fps_list), 30)
            
            # Add FPS to image
            cv2.putText(result_frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the result
            cv2.imshow('Chair Detection', result_frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance stats
        if fps_list:
            print(f"Average FPS: {sum(fps_list) / len(fps_list):.2f}")
            print(f"Min FPS: {min(fps_list):.2f}")
            print(f"Max FPS: {max(fps_list):.2f}")

if __name__ == "__main__":
    main()
```

#### c. Image Detection Script (`image_detection.py`)
```python
import cv2
import time
import argparse
import numpy as np
import os
from chair_detector import ChairDetector

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Detect chairs in images using FPGA accelerator")
    parser.add_argument("--input", required=True, help="Path to input image or directory of images")
    parser.add_argument("--output", help="Path to save output images (default: ./output)")
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output if args.output else "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = ChairDetector()
    
    # Process input (single image or directory)
    if os.path.isdir(args.input):
        # Process all images in directory
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(args.input) if os.path.splitext(f.lower())[1] in image_exts]
        
        for img_file in image_files:
            process_image(os.path.join(args.input, img_file), 
                          os.path.join(output_dir, img_file),
                          detector)
    else:
        # Process single image
        output_path = os.path.join(output_dir, os.path.basename(args.input))
        process_image(args.input, output_path, detector)

def process_image(input_path, output_path, detector):
    print(f"Processing {input_path}...")
    
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    # Start timing
    start_time = time.time()
    
    # Detect chairs
    boxes, scores = detector.detect(image)
    
    # Calculate inference time
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    
    # Draw results
    result_image = detector.draw_detections(image, boxes, scores)
    
    # Add inference time to image
    cv2.putText(result_image, f"Time: {inference_time:.4f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save output
    cv2.imwrite(output_path, result_image)
    print(f"Saved result to {output_path}")
    print(f"Detected {len(boxes)} chairs")

if __name__ == "__main__":
    main()
```

### 3. Jupyter Notebook Interface
Create a Jupyter notebook for interactive demonstration:
```bash
# Create a notebook file
touch ~/chair_detector/Chair_Detection_Demo.ipynb
```

Add the following content to the notebook:
```python
# Chair Detection Demo
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
from chair_detector import ChairDetector

# Initialize the detector
detector = ChairDetector()

# Function to process images and display results
def detect_chairs(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect chairs
    boxes, scores = detector.detect(img)
    
    # Draw results
    result_img = img_rgb.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        
        # Draw rectangle
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add label
        label = f"Chair: {score:.2f}"
        cv2.putText(result_img, label, (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display results
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Detected Chairs")
    plt.imshow(result_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return len(boxes)

# Example usage
sample_image = "sample_chair.jpg"  # Update with actual image path
num_chairs = detect_chairs(sample_image)
print(f"Detected {num_chairs} chairs in the image")
```

## Testing and Evaluation

### Performance Metrics
Evaluate the system using the following metrics:
1. **Accuracy**: 
   - Precision, recall, and F1-score on test images
   - Comparison with ground truth annotations

2. **Speed**:
   - Frames per second (FPS) for different image resolutions
   - Latency (time per image)

3. **Power Consumption**:
   - Power usage during inference
   - Comparison with CPU-based inference

### Optimizations
If performance is not satisfactory, consider:
1. Reducing input resolution
2. Adjusting confidence threshold
3. Optimizing pre/post-processing operations
4. Fine-tuning the hardware parameters

## Troubleshooting

### Common Issues
1. **Bitstream Loading Errors**:
   - Ensure the `.bit` and `.hwh` files are in the same directory
   - Check that file permissions are correct

2. **Performance Issues**:
   - Reduce image resolution for faster processing
   - Optimize pre/post-processing code
   - Consider running pre/post-processing on the ARM CPU in parallel

3. **Detection Quality Issues**:
   - Adjust confidence and NMS thresholds
   - Ensure proper image preprocessing
   - Consider re-training or fine-tuning the model

## Demo Application

Create a simple web-based interface using Flask for demonstrating the chair detection system:
```bash
# Install Flask
pip3 install flask

# Create a basic web server
touch ~/chair_detector/app.py
```

Add the following code to `app.py`:
```python
from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
import os
import time
import uuid
from chair_detector import ChairDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULT_FOLDER'] = './results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Initialize detector
detector = ChairDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Generate unique filename
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    
    # Save uploaded file
    file.save(file_path)
    
    # Read image
    image = cv2.imread(file_path)
    
    # Start timing
    start_time = time.time()
    
    # Detect chairs
    boxes, scores = detector.detect(image)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Draw detections
    result_img = detector.draw_detections(image, boxes, scores)
    
    # Add timing information
    cv2.putText(result_img, f"Time: {process_time:.4f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save result
    cv2.imwrite(result_path, result_img)
    
    return jsonify({
        'result_file': filename,
        'detection_count': len(boxes),
        'process_time': process_time
    })

@app.route('/results/<filename>')
def get_result(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

Create a basic HTML template:
```bash
mkdir -p ~/chair_detector/templates
touch ~/chair_detector/templates/index.html
```

Add the following content to `index.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Chair Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .result-container {
            margin-top: 20px;
        }
        img {
            max-width: 100%;
            margin-top: 10px;
        }
        .stats {
            margin-top: 10px;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>FPGA-Accelerated Chair Detector</h1>
    <p>Upload an image to detect chairs using the PYNQ Z2 FPGA accelerator.</p>
    
    <form id="upload-form" enctype="multipart/form-data">
        <input type="file" id="image-upload" name="image" accept="image/*">
        <button type="submit">Detect Chairs</button>
    </form>
    
    <div class="result-container" id="result-container" style="display: none;">
        <h2>Detection Result</h2>
        <div class="stats" id="stats"></div>
        <img id="result-image" src="">
    </div>
    
    <script>
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            var formData = new FormData();
            var fileInput = document.getElementById('image-upload');
            
            if (fileInput.files.length === 0) {
                alert('Please select an image to upload');
                return;
            }
            
            formData.append('image', fileInput.files[0]);
            
            fetch('/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('stats').innerHTML = 
                    `<p><strong>Chairs detected:</strong> ${data.detection_count}</p>
                     <p><strong>Processing time:</strong> ${data.process_time.toFixed(4)} seconds</p>`;
                
                document.getElementById('result-image').src = `/results/${data.result_file}`;
                document.getElementById('result-container').style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during processing');
            });
        });
    </script>
</body>
</html>
```

## Running the Demo Application
```bash
cd ~/chair_detector
python3 app.py
```

Access the web interface at `http://<PYNQ_IP>:8080`.