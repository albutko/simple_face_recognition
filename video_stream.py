import cv2
import threading
import time

class VideoStream:
    def __init__(self, src=0):
        self.vc = cv2.VideoCapture(index=src)
        self._thread = threading.Thread(name='video_capture', target=self._update_frame, daemon=True)
        (self.retval, self.frame) = self.vc.read()
        self.stopped = False

    def start(self):
        self._thread.start()
        time.sleep(2)

    def read(self):
        return self.frame

    def close(self):
        self.vc.release()
        self.stopped = True

    def _update_frame(self):
        while True:
            retval, frame = self.vc.read()
            self.frame = frame
            if self.stopped:
                break
    def isOpened(self):
        return self.vc.isOpened()

