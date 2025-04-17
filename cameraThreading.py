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
import threading
from queue import Queue
from gtts import gTTS
from ultralytics import YOLO

# Thread-safe queue and stop flag
frame_queue = Queue(maxsize=5)
stop_event = threading.Event()

# Audio text-to-speech
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
    except Exception as e:
        print(f"TTS Error: {e}")

# Function to query Ollama model for instructions
def query_ollama(prompt, model="gemma3:1b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            print(f"Ollama error {response.status_code}")
            return "Move away from danger."
    except Exception as e:
        print(f"Ollama query error: {e}")
        return "Move away from danger."

def get_avoidance_instructions(object_name, zone):
    prompt = f"""
    You are a bicycle safety assistant. A {object_name} has been detected in the following zone:
    {zone}
    Provide a short, clear instruction (10 words or less).
    """
    return query_ollama(prompt)

# Detection + warning thread
def detection_warning_thread(model):
    relevant_classes = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    frame_counter = 0
    DETECTION_FREQUENCY = 8
    cooldown = 3
    last_announcement_time = 0

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        frame_counter += 1

        if frame_counter % DETECTION_FREQUENCY != 0:
            continue

        height, width = frame.shape[:2]
        try:
            results = model(frame, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if hasattr(results[0], 'boxes') else []
        except Exception as e:
            print(f"Inference error: {e}")
            continue

        warnings = []
        for idx, box in enumerate(boxes):
            try:
                class_id = int(results[0].boxes.cls[idx].item())
                if class_id not in relevant_classes:
                    continue

                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2

                zone = "middle"
                if center_x < width / 3:
                    zone = "left"
                elif center_x > 2 * width / 3:
                    zone = "right"

                class_name = relevant_classes[class_id]
                print(f"{class_name} detected in {zone} zone")
                instruction = get_avoidance_instructions(class_name, zone)
                warnings.append((class_name, instruction))
            except Exception as e:
                print(f"Detection processing error: {e}")

        if warnings and time.time() - last_announcement_time > cooldown:
            for warning in warnings:
                speak_text(f"Warning: {warning[0]} detected. {warning[1]}")
            last_announcement_time = time.time()

# Singing thread
def singing_thread():
    lyrics = [
        "Keep riding safe and sound.",
        "Look out left, look out right.",
        "Danger zones are in your sight.",
        "Ride along, stay alert.",
        "Protect yourself, don’t get hurt."
    ]
    while not stop_event.is_set():
        for line in lyrics:
            speak_text(line)
            time.sleep(2)
        time.sleep(5)  # Pause between verses

# --- Main Execution ---
if __name__ == "__main__":
    try:
        model = YOLO("yolov8n.pt")
        torch.backends.cudnn.benchmark = True
    except Exception as e:
        print(f"YOLO load error: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    time.sleep(1)

    if not cap.isOpened():
        print("Camera not found")
        sys.exit(1)

    # Start both threads
    detection_thread = threading.Thread(target=detection_warning_thread, args=(model,))
    music_thread = threading.Thread(target=singing_thread)

    detection_thread.start()
    music_thread.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            if not frame_queue.full():
                frame_queue.put(frame)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stop_event.set()
        detection_thread.join()
        music_thread.join()
        cap.release()
        print("Resources released")
