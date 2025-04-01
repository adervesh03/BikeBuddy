# main.py
import cv2
import time
import vision
from logic import get_avoidance_instructions
from alerts import speak_text

GRID_ROWS, GRID_COLS = 6, 6
DETECTION_FREQUENCY = 8
announcement_cooldown = 3
last_announcement_time = 0
frame_counter = 0

try:
    while vision.cap.isOpened():
        ret, frame = vision.cap.read()
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
                results = vision.model(frame, verbose=False)
                detected_objects = results[0].boxes.xyxy.cpu().numpy().tolist() if hasattr(results[0], 'boxes') else []
            except Exception as e:
                print(f"Inference error: {e}")
                continue

            warnings = []
            for idx, box in enumerate(detected_objects):
                try:
                    class_id = int(results[0].boxes.cls[idx].item())
                    if class_id not in vision.relevant_classes:
                        continue

                    x1, y1, x2, y2 = box
                    zone = vision.get_zone((x1, y1, x2, y2), frame.shape)
                    class_name = vision.relevant_classes[class_id]

                    print(f"{class_name} detected in zone: {zone}")

                    instruction = get_avoidance_instructions(class_name, zone)
                    warnings.append((class_name, instruction))

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
    vision.cap.release()
    cv2.destroyAllWindows()
    print("Resources released")
