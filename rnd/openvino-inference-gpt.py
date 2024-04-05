from openvino.runtime import Core
import cv2
import numpy as np

# Initialize OpenVINO Core
ie = Core()

# Path to the XML and BIN files of the OpenVINO IR model
model_xml = "models/best_openvino_model/best.xml"
model_bin = "models/best_openvino_model/best.bin"

# Read and load the model
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Load an image
image = cv2.imread("source_files\construction-safety.jpg")
input_image = cv2.resize(image, (640, 640))
input_image = input_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension


# Assuming your model has only one input and one output
input_key = next(iter(compiled_model.inputs))
output_key = next(iter(compiled_model.outputs))

# Perform inference
result = compiled_model([input_image])[output_key]

print("Result shape:", result.shape)
print("Result data:", result)
