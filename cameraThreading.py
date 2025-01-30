import cv2
import threading 
import queue

class CameraThread:
    def __init__(self, cameraIndices, queue_size=10): 
        self.cameraIndices = cameraIndices
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.threads = []
        self.running = True
        
    def capture_frames(self, cameraIndex): 
        cap = cv2.VideoCapture(cameraIndex)
        
        while self.running: 
            ret, frame = cap.read()
            if ret:
                self.frame_queue.put((cameraIndex, frame), timeout=1)
            else: 
                print(f'Error reading frame from camera {cameraIndex}')
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
        return self.frame_queue 