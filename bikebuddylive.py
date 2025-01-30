import cv2
import numpy as np
import pandas as pd
import yolov5
import ollama
import time
import warnings
import threading
import queue

warnings.filterwarnings('ignore', category=FutureWarning)

# Create queues for thread communication
description_queue = queue.Queue(maxsize=1)  # Only store latest description
response_queue = queue.Queue()

def ai_thread():
    """Thread function for AI processing"""
    while True:
        try:
            description = description_queue.get(timeout=1.0)
            response = ollama.generate('deepseek-r1:1.5b', description)
            response_queue.put(response['response'])
            description_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"AI thread error: {e}")

# Start AI thread
ai_thread = threading.Thread(target=ai_thread, daemon=True)
ai_thread.start()

model = yolov5.load('yolov5s.pt')

# Initialize webcam
cap = cv2.VideoCapture("bikeVideo.mov")  # Use default webcam (0)

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
        if score > 0.3:  # Confidence threshold
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
            description += f"{results.names[int(category)]} at ({x1}, {y1}, {x2}, {y2}) with confidence {score:.2f}\n"
    
    # Pass frame to ollama model once every second
    if current_time - last_inference_time >= 1:
        description += "Notify riders of their surroundings. You are provided a description of your surroundings and if there are any cars, people, bikes, and motorcycles around you."
        description += "Generate a short one sentence summary notifying the rider of each object around them. For example: There is a bicycle on your left, or there is a car on your right, or there is a person in front of you."
        description += "If there are no objects of a category detected, do not include that category in the summary."
        
        # Only add if queue is empty to avoid backlog
        if description_queue.empty():
            description_queue.put(description)
        last_inference_time = current_time

    # Check for AI responses
    try:
        response = response_queue.get_nowait()
        print(response)
        response_queue.task_done()
    except queue.Empty:
        pass
    
    # Show frame
    cv2.imshow('YOLOv5 Detection', frame)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()