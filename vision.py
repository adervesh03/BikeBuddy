# vision.py
import cv2
import torch
from ultralytics import YOLO

# Load YOLO model
try:
    model = YOLO('yolov8n.pt')
    torch.backends.cudnn.benchmark = True
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Initialize camera
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit(1)
except Exception as e:
    print(f"Camera initialization error: {e}")
    exit(1)

# Relevant object classes for detection
relevant_classes = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Determine left/middle/right zone based on bounding box center
def get_zone(bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    width = frame_shape[1]
    center_x = (x1 + x2) / 2

    if center_x < width / 3:
        return "left"
    elif center_x > 2 * width / 3:
        return "right"
    else:
        return "middle"
