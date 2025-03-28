import cv2
import torch
import time
import os
import sys
import subprocess
import tempfile
import numpy as np
import requests
import json
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

# Function to query Ollama model for instructions
def query_ollama(prompt, model="gemma3:1b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return "Move away from danger."
    except Exception as e:
        print(f"Ollama query error: {e}")
        return "Move away from danger."

# Function to get LLM-generated instructions based on grid information
def get_avoidance_instructions(object_name, grid_info):
    prompt = f"""
    You are a bicycle safety assistant. A {object_name} has been detected in the following zone:
    {grid_info}
    
    Provide a short, clear instruction (10 words or less) to help the cyclist avoid the {object_name}.
    """
    return query_ollama(prompt)

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
                    
                    width = frame.shape[1]

                    center_x = (int(x1 + x2) / 2)
                    center_y = (int(y1 + y2) / 2)

                    zone = "middle"

                    if center_x < width / 3:
                        zone = "left"
                    elif center_x > 2 * width / 3:
                        zone = "right"
                    else:
                        zone = "middle"

                    grid_cells = [(r, c) for r in range(start_row, end_row + 1) for c in range(start_col, end_col + 1)]
                    class_name = relevant_classes[class_id]
                    
                    print(f"{class_name} detected in zone: {zone}")
                    
                    # Use LLM to generate avoidance instructions
                    instruction = get_avoidance_instructions(class_name, zone)
                    warnings.append((class_name, instruction))
                    
                    '''
                    for cell in grid_cells:
                        if cell in WARNING_ZONE:
                            warnings.append((class_name, "Move left or right!"))
                        elif cell in LEFT_ZONE:
                            warnings.append((class_name, "Move right!"))
                        elif cell in RIGHT_ZONE:
                            warnings.append((class_name, "Move left!"))
                    '''

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
