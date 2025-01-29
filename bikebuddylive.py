import cv2
import numpy as np
import pandas as pd
import yolov5
import ollama
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

model = yolov5.load('yolov5s.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use default webcam (0)

# initialize variables
last_inference_time = time.time()
frame_count = 0
people_detected = 0
bicycles_detected = 0
cars_detected = 0
motorcycles_detected = 0

while True:
    # Read frame
    ret, frame = cap.read()

    # update frame count
    

    if not ret:
        break
        
    # Run YOLOv5 inference
    results = model(frame)
    
    # Get detection data
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # update object count
    # get the number of people detected
    people_detected = len([category for category in categories if category == 0])
    # get the number of bicycles detected
    bicycles_detected = len([category for category in categories if category == 1])
    # get the number of cars detected
    cars_detected = len([category for category in categories if category == 2])
    # get the number of motorcycles detected
    motorcycles_detected = len([category for category in categories if category == 3])

    user_prompt = f"Number of people detected: {people_detected}\nNumber of bicycles detected: {bicycles_detected}\nNumber of cars detected: {cars_detected}\nNumber of motorcycles detected: {motorcycles_detected}"
    cv2.putText(frame, user_prompt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
        
        # Pass text to ollama model
        response = ollama.generate('deepseek-r1:1.5b', description)
        #cv2.putText(frame, response, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        print(response['response'])
        last_inference_time = current_time
    
    # Show frame
    cv2.imshow('YOLOv5 Detection', frame)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()