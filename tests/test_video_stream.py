import cv2
from video_stream import VideoStream
import time

if __name__ == '__main__':

    s = VideoStream()
    s.start()
    start_time = time.time()

    while time.time() - start_time < 10:
       f = s.get_frame()
       cv2.imshow('frame', f)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    s.close()



