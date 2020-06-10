import cv2
import threading
import time

class VideoStream:
    def __init__(self, src=0):
        self.vc = cv2.VideoCapture(index=src)
        (self.retval, self.frame) = self.vc.read()
        self.stopped = False

    def start(self):
        thread = threading.Thread(target=self._update_frame, daemon=True)
        thread.start()

    def read(self):
        return self.frame

    def close(self):
        self.vc.release()
        self.stopped = True

    def _update_frame(self):
        while True:
            if self.stopped:
                return
            retval, frame = self.vc.read()
            self.frame = frame

    def isOpened(self):
        return self.vc.isOpened()

