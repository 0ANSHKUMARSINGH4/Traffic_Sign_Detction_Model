
from ultralytics import YOLO
import cv2

# Load the Universal Model
model = YOLO('yolov8l-worldv2.pt')

# Define Custom Vocabulary (Indian Context)
model.set_classes([
    "cow", "pothole", "auto rickshaw", "speed bump", 
    "traffic sign", "helmet", "person", "vehicle", "bus", "truck"
])

# Run Inference
# Replace 'input.jpg' with your image path
results = model.predict('input.jpg', augment=True, conf=0.15)
results[0].save('output.jpg')
print("✅ Detection saved to output.jpg")
