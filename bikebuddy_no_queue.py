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
            
            # determine if the object is in the left, right, or middle zone
            width = frame.shape[1]
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if center_x < width / 3:
                zone = "left"
            elif center_x > 2 * width / 3:
                zone = "right"
            else:
                zone = "middle"
            
            # update text description of objects in frame
            description += f"{results.names[int(category)]} is in zone ({zone}) at ({x1}, {y1}, {x2}, {y2})\n"

    # Pass frame to ollama model once every second
    if current_time - last_inference_time >= 1:
        print(description)
        description += "Tell me how to avoid the object based on its position. Nothing more, nothing less."
        
        response = ollama.generate('gemma3:4b', description)
        print(f"Response:\n{response['response']}\n")
        last_inference_time = current_time
    
    # Show frame
    cv2.imshow('YOLOv5 Detection', frame)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()