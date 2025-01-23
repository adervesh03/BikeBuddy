import cv2
import numpy as np
import pandas as pd
import yolov5

model = yolov5.load('yolov5s.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use default webcam (0)

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
    
    # Draw boxes
    for box, score, category in zip(boxes, scores, categories):
        if score > 0.3:  # Confidence threshold
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            label = f'{results.names[int(category)]}: {score:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('YOLOv5 Detection', frame)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()