# APD/PPE Detection System with OpenVINO Optimization

## Description
The APD/PPE Detection System utilizes transfer learning on YOLOv8 for efficient detection of APDs (Abnormal Personal Protective Equipment) or PPE (Personal Protective Equipment). The system leverages the power of deep learning to detect abnormalities in protective gear, vital for maintaining safety standards in various environments.

Additionally, the system employs OpenVINO for quantization, enhancing performance by optimizing the model to perform inference up to 3 times faster. This optimization ensures real-time or near-real-time processing, crucial for swift response to potential hazards.

## Installation
1. Install the required dependencies using the following command:
    ```
    pip install -r requirements.txt
    ```
   This command will ensure that all necessary libraries are installed, including Torch, OpenCV, Flask, and Gunicorn.

## Execution
1. Run the `apdDetect.py` script. This script initializes the APD/PPE detection system and prepares it for inference.

## Accessing the Output
1. Open your localhost to observe the output on a streaming website. Simply click on your running localhost to access the streaming service. The website will display real-time or near-real-time detections of APDs or PPE anomalies.

By following these steps, users can efficiently deploy and utilize the APD/PPE Detection System to enhance safety measures and ensure compliance with protective equipment standards.
