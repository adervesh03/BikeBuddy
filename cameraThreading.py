import cv2
import threading 
import queue

class CameraThread:
    def __init__(self, cameraIndices, queue_size=500): 
        self.cameraIndices = cameraIndices
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.threads = []
        self.running = True
        
    def capture_frames(self, cameraIndex): 
        cap = cv2.VideoCapture(cameraIndex)
        if not cap.isOpened():
            print(f"Error: Camera {cameraIndex} could not be opened.")
            return
        
        while self.running: 
            ret, frame = cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()  # Remove the oldest frame to avoid blocking
                self.frame_queue.put(frame)  # Add the new frame
            else: 
                print(f"Error reading frame from camera {cameraIndex}")
                break
        
        cap.release()
        print(f'Camera {cameraIndex} capture stopped')
        
    def startCameras(self): 
        for cameraIndex in self.cameraIndices:
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
            return self.frame_queue.get()  # Return the next available frame
        return None  # Return None if no frame is available
 