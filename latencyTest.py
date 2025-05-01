#!/usr/bin/env python3

import cv2
import torch
import time
import os
import sys
import subprocess
import tempfile
from datetime import datetime
from ultralytics import YOLO
from gtts import gTTS

def speak_text(text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            temp_filename = tmp_file.name
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_filename)
        
        try:
            subprocess.call(['mpg123', '-q', temp_filename])
        except FileNotFoundError:
            try:
                subprocess.call(['aplay', temp_filename])
            except FileNotFoundError:
                if sys.platform == "darwin":
                    subprocess.call(['afplay', temp_filename])
                elif sys.platform == "win32":
                    os.startfile(temp_filename)
        
        os.remove(temp_filename)
        return True
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        return False

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

MAX_DETECTIONS = 5
detection_count = 0
current_detection = None

# Lists to store detection data
detection_times = []
warning_times = []
detected_classes = []
detected_boxes = []
latencies = []

log_filename = f"detection_log_{timestamp}.txt"
with open(log_filename, "w") as log_file:
    log_file.write("Latency Test Log\n")
    log_file.write(f"Test started at: {get_timestamp()}\n")
    log_file.write(f"Target detections: {MAX_DETECTIONS}\n\n")

print(f"Starting latency test. Recording to {output_file}")
print(f"Logging timestamps to {log_filename}")
print(f"Will perform {MAX_DETECTIONS} detections before stopping")

try:
    while cap.isOpened() and detection_count < MAX_DETECTIONS:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Can't receive frame")
            break
        
        current_time = get_timestamp()
        
        if current_detection is None:
            try:
                results = model(frame, verbose=False)
                
                if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    for idx, box in enumerate(results[0].boxes.xyxy.cpu().numpy().tolist()):
                        class_id = int(results[0].boxes.cls[idx].item())
                        if class_id in relevant_classes:
                            detection_time = get_timestamp()
                            detected_class = relevant_classes[class_id]
                            current_detection = detection_count + 1
                            
                            detection_times.append(detection_time)
                            detected_classes.append(detected_class)
                            detected_boxes.append(box)
                            
                            with open(log_filename, "a") as log_file:
                                log_file.write(f"Detection #{current_detection}\n")
                                log_file.write(f"Detection time: {detection_time} - {detected_class}\n")
                            
                            print(f"Detection #{current_detection}: {detected_class} at {detection_time}")
                            break
            except Exception as e:
                print(f"Inference error: {e}")
        
        elif len(warning_times) < len(detection_times):
            detection_index = len(warning_times)
            warning_message = f"Warning: {detected_classes[detection_index]} detected."
            print(f"Speaking: {warning_message}")
            speak_text(warning_message)
            
            warning_time = get_timestamp()
            warning_times.append(warning_time)
            
            start_time = datetime.strptime(detection_times[detection_index], "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(warning_time, "%Y-%m-%d %H:%M:%S.%f")
            latency = end_time - start_time
            latencies.append(latency.total_seconds())
            
            with open(log_filename, "a") as log_file:
                log_file.write(f"Warning time: {warning_time}\n")
                log_file.write(f"Detection to warning latency: {latency.total_seconds()} seconds\n\n")
            
            print(f"Warning #{current_detection} provided at {warning_time}")
            print(f"Latency: {latency.total_seconds()} seconds")
            
            detection_count += 1
            if detection_count < MAX_DETECTIONS:
                current_detection = None
        
        out.write(frame)

except KeyboardInterrupt:
    print("Program interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    out.release()
    
    with open(log_filename, "a") as log_file:
        log_file.write(f"Test completed at: {get_timestamp()}\n")
        log_file.write(f"Total detections completed: {detection_count}/{MAX_DETECTIONS}\n\n")
        
        if detection_times and warning_times:
            log_file.write("Summary of latencies:\n")
            for i in range(len(latencies)):
                log_file.write(f"Detection #{i+1}: {latencies[i]:.3f} seconds\n")
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                log_file.write(f"\nAverage latency: {avg_latency:.3f} seconds\n")
        
        log_file.write(f"\nVideo saved to: {output_file}\n")
    
    if detection_count > 0:
        print(f"Test completed with {detection_count}/{MAX_DETECTIONS} detections!")
        print(f"Video saved to: {output_file}")
        print(f"Log saved to: {log_filename}")
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"Average latency: {avg_latency:.3f} seconds")
    else:
        print("Test completed without any detections.")
