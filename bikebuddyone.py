import cv2
import torch
import time
import os
import sys
import subprocess
import tempfile
import numpy as np
from gtts import gTTS
from ultralytics import YOLO

# Function to provide audio warnings
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
        
        time.sleep(1)
        os.remove(temp_filename)
    except Exception as e:
        print(f"Text-to-speech error: {e}")

# Load YOLO model
try:
    model = YOLO('yolov8n.pt')
    torch.backends.cudnn.benchmark = True
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
except Exception as e:
    print(f"Camera initialization error: {e}")
    sys.exit(1)

# Grid settings
GRID_ROWS, GRID_COLS = 6, 6
WARNING_ZONE = [(4, 2), (4, 3), (5, 2), (5, 3)]  # Center-bottom area of the grid
LEFT_ZONE = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS // 2)]
RIGHT_ZONE = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS // 2, GRID_COLS)]

# Object detection settings
relevant_classes = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
frame_counter = 0
DETECTION_FREQUENCY = 8
announcement_cooldown = 3
last_announcement = ""
last_announcement_time = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Can't receive frame")
            break
        if frame.size == 0:
            print("Error: Empty frame received")
            continue
        
        frame_counter += 1
        if frame_counter % DETECTION_FREQUENCY == 0:
            height, width = frame.shape[:2]
            cell_width, cell_height = width / GRID_COLS, height / GRID_ROWS

            try:
                results = model(frame, verbose=False)
                detected_objects = results[0].boxes.xyxy.cpu().numpy().tolist() if hasattr(results[0], 'boxes') else []
            except Exception as e:
                print(f"Inference error: {e}")
                continue
            
            warnings = []
            
            for idx, box in enumerate(detected_objects):
                try:
                    class_id = int(results[0].boxes.cls[idx].item())
                    if class_id not in relevant_classes:
                        continue
                    
                    x1, y1, x2, y2 = box
                    start_col, end_col = int(x1 / cell_width), int(x2 / cell_width)
                    start_row, end_row = int(y1 / cell_height), int(y2 / cell_height)
                    grid_cells = [(r, c) for r in range(start_row, end_row + 1) for c in range(start_col, end_col + 1)]
                    class_name = relevant_classes[class_id]
                    
                    print(f"{class_name} detected in grid cells: {grid_cells}")
                    
                    for cell in grid_cells:
                        if cell in WARNING_ZONE:
                            warnings.append((class_name, "Move left or right!"))
                        elif cell in LEFT_ZONE:
                            warnings.append((class_name, "Move right!"))
                        elif cell in RIGHT_ZONE:
                            warnings.append((class_name, "Move left!"))
                except Exception as e:
                    print(f"Error processing detection {idx}: {e}")
            
            if warnings:
                current_time = time.time()
                if current_time - last_announcement_time > announcement_cooldown:
                    for warning in warnings:
                        speak_text(f"Warning: {warning[0]} detected. {warning[1]}")
                    last_announcement_time = current_time
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program interrupted by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released")
