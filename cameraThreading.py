import threading
import queue
import time
import cv2

class CameraThread:
    def __init__(self, cameraIndex):
        self.cameraIndex = cameraIndex
        self.frame_queue = queue.Queue()
        self.stop_event = threading.Event()  # Used to stop the loop
        self.threads = []

    def startCameras(self):
        thread = threading.Thread(target=self.capture_frames, args=(self.cameraIndex,))
        thread.start()
        self.threads.append(thread)

    def stopCameras(self):
        print("Stopping cameras...")
        self.stop_event.set()  # Signal all threads to stop
        for thread in self.threads:
            thread.join()  # Wait for them to exit
        print("Cameras stopped.")

    def capture_frames(self, cameraIndex):
        print(f"Thread started for camera {cameraIndex}")
        cap = cv2.VideoCapture(cameraIndex)  # Open the camera

        if not cap.isOpened():
            print(f"ERROR: Camera {cameraIndex} failed to open")
            return

        while not self.stop_event.is_set():  # Stop when event is set
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to grab frame!")
                continue

            print(f"Frame captured from camera {cameraIndex}, type: {type(frame)}")
            self.frame_queue.put(frame)
            time.sleep(1)  # Simulate a 1-second frame rate

        print(f"Thread for camera {cameraIndex} stopping.")
        cap.release()  # Release the camera when stopping

        
    def startCameras(self): 
        for cameraIndex in self.cameraIndex:
            thread = threading.Thread(target=self.capture_frames, args=(cameraIndex,))
            self.threads.append(thread)
            thread.start()
            
        print('Cameras started')
        
    def stopCameras(self): 
        self.running = False
        for thread in self.threads:
            thread.join()
        print('Cameras stopped')
        
    def getFrames(self):
        """Retrieve the latest frame from the queue if available."""
        if not self.frame_queue.empty():
            item = self.frame_queue.get()
            print(f"Retrived frame {item} from queue")
            return item  # Return the next available frame
        return None  # Return None if no frame is available
 