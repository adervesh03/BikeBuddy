#!/usr/bin/env python3

import cv2
import torch
import time
import os
import sys
from datetime import datetime
from ultralytics import YOLO

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# Load YOLO model
try:
    model = YOLO('yolov8n.pt')
    torch.backends.cudnn.benchmark = True
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

# Initialize camera
try:
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    print("Camera initialized successfully")
except Exception as e:
    print(f"Camera initialization error: {e}")
    sys.exit(1)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

# Initialize video writer
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"detection_{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Object detection settings
relevant_classes = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Duration for recording (25 seconds)
DURATION = 25  # in seconds

# Variables to track detection time
start_time = time.time()
end_time = start_time + DURATION
total_detection_time = 0
current_detection_start = None
detected_objects = {}
frame_count = 0
detection_count = 0

# Create log file
log_filename = f"detection_presence_log_{timestamp}.txt"
with open(log_filename, "w") as log_file:
    log_file.write("Object Detection Presence Log\n")
    log_file.write(f"Test started at: {get_timestamp()}\n")
    log_file.write(f"Test duration: {DURATION} seconds\n\n")

print(f"Starting object detection timer. Recording to {output_file}")
print(f"Logging timestamps to {log_filename}")
print(f"Will run for {DURATION} seconds")

try:
    while cap.isOpened() and time.time() < end_time:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Can't receive frame")
            break
        
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - start_time
        remaining = end_time - current_time
        
        # Run object detection on every frame
        objects_detected = False
        detected_classes_this_frame = []
        
        try:
            results = model(frame, verbose=False)
            
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                for idx, box in enumerate(results[0].boxes.xyxy.cpu().numpy().tolist()):
                    class_id = int(results[0].boxes.cls[idx].item())
                    confidence = float(results[0].boxes.conf[idx].item())
                    
                    if class_id in relevant_classes and confidence > 0.5:
                        objects_detected = True
                        class_name = relevant_classes[class_id]
                        detected_classes_this_frame.append(class_name)
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Track object type frequency
                        if class_name not in detected_objects:
                            detected_objects[class_name] = 0
                        detected_objects[class_name] += 1
        
        except Exception as e:
            print(f"Inference error: {e}")
        
        # Track time when objects are detected
        if objects_detected:
            detection_count += 1
            if current_detection_start is None:
                current_detection_start = current_time
                print(f"Object detection started at elapsed time: {elapsed:.2f}s")
                with open(log_filename, "a") as log_file:
                    log_file.write(f"Detection started at: {elapsed:.2f}s - {', '.join(detected_classes_this_frame)}\n")
        else:
            if current_detection_start is not None:
                detection_duration = current_time - current_detection_start
                total_detection_time += detection_duration
                print(f"Object detection ended, duration: {detection_duration:.2f}s")
                with open(log_filename, "a") as log_file:
                    log_file.write(f"Detection ended at: {elapsed:.2f}s, duration: {detection_duration:.2f}s\n\n")
                current_detection_start = None
        
        # Add elapsed time to frame
        cv2.putText(frame, f"Time: {elapsed:.2f}s / {DURATION}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add detection status to frame
        status = "Detecting" if objects_detected else "No Detection"
        cv2.putText(frame, status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if not objects_detected else (0, 255, 0), 2)
        
        # Write frame to video
        out.write(frame)

except KeyboardInterrupt:
    print("Program interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    # If objects were being detected when the program ends, add the final detection duration
    if current_detection_start is not None:
        detection_duration = time.time() - current_detection_start
        total_detection_time += detection_duration
    
    # Calculate detection percentage
    total_elapsed = time.time() - start_time
    detection_percentage = (total_detection_time / total_elapsed) * 100
    
    cap.release()
    out.release()
    
    with open(log_filename, "a") as log_file:
        log_file.write(f"\nTest completed at: {get_timestamp()}\n")
        log_file.write(f"Total elapsed time: {total_elapsed:.2f} seconds\n")
        log_file.write(f"Total time objects detected: {total_detection_time:.2f} seconds\n")
        log_file.write(f"Detection percentage: {detection_percentage:.2f}%\n\n")
        
        if detected_objects:
            log_file.write("Detected objects summary:\n")
            for obj_class, count in detected_objects.items():
                log_file.write(f"{obj_class}: {count} frames\n")
        
        log_file.write(f"\nTotal frames: {frame_count}\n")
        log_file.write(f"Frames with detections: {detection_count}\n")
        log_file.write(f"Detection ratio: {detection_count/frame_count:.2f}\n")
        log_file.write(f"\nVideo saved to: {output_file}\n")
    
    print(f"\nTest completed!")
    print(f"Video saved to: {output_file}")
    print(f"Log saved to: {log_filename}")
    print(f"Total elapsed time: {total_elapsed:.2f} seconds")
    print(f"Total time objects detected: {total_detection_time:.2f} seconds")
    print(f"Detection percentage: {detection_percentage:.2f}%")
    
    if detected_objects:
        print("\nDetected objects summary:")
        for obj_class, count in detected_objects.items():
            print(f"{obj_class}: {count} frames")