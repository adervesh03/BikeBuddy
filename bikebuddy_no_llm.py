import cv2
import numpy as np
import pandas as pd
import yolov5
import ollama
import time
import warnings
import threading
import queue
import subprocess

warnings.filterwarnings('ignore', category=FutureWarning)

model = yolov5.load('yolov5s.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use default webcam (0)

# initialize variables
last_inference_time = time.time()

while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        break
        
    # Run YOLOv5 inference
    results = model(frame)
    
    # Get detection data
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    current_time = time.time()
    description = "Detected objects:\n"
    
    for box, score, category in zip(boxes, scores, categories):
        if score > 0.6:  # Confidence threshold
            # draw bounding boxes
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            label = f'{results.names[int(category)]}: {score:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # update text description of objects in frame
            x1, y1, x2, y2 = box
            print(f"{results.names[int(category)]} at ({x1}, {y1}, {x2}, {y2}) with confidence {score:.2f}")
            # description += f"{results.names[int(category)]} at ({x1}, {y1}, {x2}, {y2}) with confidence {score:.2f}\n"

    # Pass frame to ollama model once every second
    if current_time - last_inference_time >= 1:
        description += "Based on only the objects provided, generate a short singular sentence for each object."
        # description += "The bounding boxes and coordinates do not represent absolute position, but are relative to the frame's dimensions. The confidence score indicates the model's confidence in the object's classification."
        # description += "You should use this information knowing that the position of each object is relative to the rider's position and orientation."
        # description += "You are to generate a short one sentence summary notifying the rider of each object around them. For example: There is a bicycle on your left, or there is a car on your right, or there is a person in front of you."
        # description += "If there are no objects of a category detected, do not include that category in the summary."
        # description += "Your answer needs to be short, concise, and limited to one sentence as if they were spoken. Do not include any explanations of the objects or any additional information."
        
        last_inference_time = current_time
    
    # Show frame
    cv2.imshow('YOLOv5 Detection', frame)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()