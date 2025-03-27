import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import time
from gtts import gTTS
import tempfile
import numpy as np
import sys
import subprocess  # For playing audio files with system commands

# Function to speak detected objects
def speak_text(text):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            temp_filename = tmp_file.name
            
        # Create and save the speech audio
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_filename)
        
        # Play the audio file using system commands
        # This works on Linux systems. For other OS, you might need different commands
        try:
            # Try mpg123 first (common on Linux)
            subprocess.call(['mpg123', '-q', temp_filename])
        except FileNotFoundError:
            try:
                # Try aplay if mpg123 is not available
                subprocess.call(['aplay', temp_filename])
            except FileNotFoundError:
                # Fallback to default system player
                if sys.platform == "darwin":  # macOS
                    subprocess.call(['afplay', temp_filename])
                elif sys.platform == "win32":  # Windows
                    os.startfile(temp_filename)
        
        # Clean up the temporary file after a short delay to ensure playback completes
        time.sleep(1)
        os.remove(temp_filename)
        
    except Exception as e:
        print(f"Text-to-speech error: {e}")

# Add error handling for model loading
try:
    # Load YOLOv8 model with memory optimization
    model = YOLO('yolov8n.pt')
    # Set model parameters for better performance
    torch.backends.cudnn.benchmark = True  # Speed up if using GPU
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

# Initialize camera with error handling
try:
    cap = cv2.VideoCapture(0)
    # Give camera time to initialize
    time.sleep(1)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
except Exception as e:
    print(f"Camera initialization error: {e}")
    sys.exit(1)

# Define grid dimensions
GRID_ROWS = 6
GRID_COLS = 6

# Keep track of recently announced objects to avoid repetition
last_announcement = ""
last_announcement_time = 0
announcement_cooldown = 3  # seconds

try:
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Can't receive frame")
                break
                
            # Check frame validity
            if frame.size == 0:
                print("Error: Empty frame received")
                continue
                
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Calculate cell dimensions
            cell_width = width / GRID_COLS
            cell_height = height / GRID_ROWS

            # Run YOLO inference with error handling
            try:
                results = model(frame, verbose=False)
                detected_objects = []
                
                # Check if results contain boxes
                if hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'xyxy'):
                    detected_objects = results[0].boxes.xyxy.cpu().numpy().tolist()
                print(f"Detected {len(detected_objects)} objects")
            except Exception as e:
                print(f"Inference error: {e}")
                continue

            # Only process cars, pedestrians, and bicycles
            relevant_classes = [0, 1, 2]  # person, bicycle, car
            filtered_boxes = []
            filtered_indices = []

            for idx, box in enumerate(detected_objects):
                try:
                    class_id = int(results[0].boxes.cls[idx].item())
                    if class_id in relevant_classes:
                        filtered_boxes.append(box)
                        filtered_indices.append(idx)
                except Exception as e:
                    print(f"Error processing detection {idx}: {e}")

            # If there's a detected object in the frame pass the object to deepseek and get the description
            if filtered_boxes:
                # Process each detected object
                for idx, box in enumerate(filtered_boxes):
                    try:
                        x1, y1, x2, y2 = box
                        
                        # Calculate grid cells that contain this object
                        start_col = max(0, min(GRID_COLS - 1, int(x1 / cell_width)))
                        end_col = max(0, min(GRID_COLS - 1, int(x2 / cell_width)))
                        start_row = max(0, min(GRID_ROWS - 1, int(y1 / cell_height)))
                        end_row = max(0, min(GRID_ROWS - 1, int(y2 / cell_height)))
                        
                        # Get object class if available
                        class_id = int(results[0].boxes.cls[filtered_indices[idx]].item())
                        class_name = results[0].names[class_id]
                        
                        # Generate grid cell list
                        grid_cells = []
                        for r in range(start_row, end_row + 1):
                            for c in range(start_col, end_col + 1):
                                grid_cells.append(f"({r},{c})")
                        
                        # Print grid information
                        print(f"{class_name} detected in grid cells: {', '.join(grid_cells)}")
                        
                        # Announce the detected object with debouncing
                        current_time = time.time()
                        announcement = f"{class_name} detected"
                        if (announcement != last_announcement or 
                            current_time - last_announcement_time > announcement_cooldown):
                            speak_text(announcement)
                            last_announcement = announcement
                            last_announcement_time = current_time
                            
                    except Exception as e:
                        print(f"Error processing object {idx}: {e}")

            # Show the frame with detections if needed
            # cv2.imshow('BikeBuddy Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Frame processing error: {e}")
            continue

except KeyboardInterrupt:
    print("Program interrupted by user")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Resources released")