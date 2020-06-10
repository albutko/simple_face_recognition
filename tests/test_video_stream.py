import threading
import pytest
import numpy as np
import time
from sfd.video_stream import VideoStream


def test_thread_starts_on_start():
    video_stream = VideoStream()
    start_threads = threading.active_count()
    video_stream.start()
    current_threads = threading.active_count()
    assert start_threads + 1 == current_threads

def test_gets_new_frame():
    video_stream = VideoStream()
    video_stream.start()
    first_frame = video_stream.read()
    time.sleep(1)
    second_frame = video_stream.read()
    assert not np.array_equal(first_frame, second_frame)

def test_close_closes_video_capture():
    video_stream = VideoStream()
    video_stream.start()
    video_stream.close()
    assert not video_stream.isOpened()



