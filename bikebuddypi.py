import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize camera
cap = cv2.VideoCapture(0)

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

        # Show detection results
        annotated_frame = results[0].plot()

        # Display using Matplotlib (headless mode)
        plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.01)  # Pause briefly to allow updating
        plt.clf()  # Clear the figure for the next frame

        # Print detected objects
        print(results[0].boxes.xyxy.tolist())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
