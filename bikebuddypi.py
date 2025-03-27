import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import pyttsx3
from gtts import gTTS
import tempfile

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize text-to-speech engine
engine = pyttsx3.init()

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break

        # Run YOLO inference
        results = model(frame)

        # Print detected objects
        detected_objects = results[0].boxes.xyxy.tolist()
        print(detected_objects)

        # Convert detection results to text and speak
        detected_text = "Detected objects: " + ", ".join([f"Object at {obj}" for obj in detected_objects])

        # Use gTTS to convert text to speech
        tts = gTTS(detected_text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            os.system(f"mpg321 {fp.name}")
            os.remove(fp.name)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()