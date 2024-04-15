# APD/PPE Detection System with OpenVINO Optimization

## Description
The APD/PPE Detection System utilizes transfer learning on YOLOv8 for efficient detection of APDs (Abnormal Personal Protective Equipment) or PPE (Personal Protective Equipment). The system leverages the power of deep learning to detect abnormalities in protective gear, vital for maintaining safety standards in various environments.

Additionally, the system employs OpenVINO for quantization, enhancing performance by optimizing the model to perform inference up to 3 times faster. This optimization ensures real-time or near-real-time processing, crucial for swift response to potential hazards.

# PPE Detection using YOLOv8

This repository implements PPE (Personal Protective Equipment) detection using YOLOv8, a state-of-the-art object detection algorithm. Below, we detail the components and usage of this project.

## 1. Model Transfer Learning

Instead of training the model from scratch, we utilized transfer learning for continuous improvement and development. We leveraged a pre-trained model checkpoint from the repository [Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection) and fine-tuned it using our own dataset. The modifications included downsampling images and adding more sample cases.

- Transfer learning code: Located in the 'transferlearning' repository.
- Results: Available in the 'runs' repository.

## 2. Quantization using OpenVINO

OpenVINO (Open Visual Inference & Neural network Optimization) is a toolkit developed by Intel for deploying deep learning models across various Intel hardware platforms. It optimizes and accelerates inference on Intel CPUs, GPUs, VPUs, and FPGAs.

Quantization is a technique used to reduce the precision of the model's parameters and/or activations to improve efficiency without significantly sacrificing accuracy. We employed OpenVINO for quantizing the model, thereby optimizing its performance.

## 3. Model Inference

Model inference is facilitated through the `inferencevideo-app.py` script. Simply execute this code, and it will initiate a server running on `localhost:5000`. Users can then access this port to perform inference on videos.

## Setup Instructions

### 1. Setting up Virtual Environment

```bash
# Create a virtual environment
python3 -m venv ppe_detection_env

# Activate the virtual environment
source ppe_detection_env/bin/activate
```

### 2. Install Required Packages

```bash
# Navigate to the project directory
cd path/to/apd-ppe-detection

# Install requirements
pip install -r requirements.txt
```

### 3. Run Inference Server

```bash
# Run the inference server
python appDetect.py
```

### 4. Access the Application

Once the server is running, open your web browser and go to `localhost:5000` to access the application for PPE detection on videos.

For further instructions and detailed usage, please refer to the documentation within each respective repository.

Feel free to reach out if you have any questions or require assistance. Happy detecting!
