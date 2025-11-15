from threading import Thread
import cv2
import time


class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)

        # Set resolution at the source for speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        self.width = width
        self.height = height

        self.ret, frame = self.cap.read()
        if self.ret:
            self.frame = cv2.resize(frame, (width, height))
        else:
            self.frame = None

        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Resize at capture for consistent speed
                    self.ret = True
                    self.frame = cv2.resize(frame, (640, 480))
                else:
                    self.ret = False
                    time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] Camera read failed: {e}")
                self.ret = False
                time.sleep(0.1)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()