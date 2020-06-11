import pytest
import numpy as np
from sfd.detector.base import BaseDetector

@pytest.fixture
def image():
    return np.random.randint(0,255,(300,300,3))

def test_detect_face_not_implemented(image):
    with pytest.raises(NotImplementedError):
        det = BaseDetector()
        det.detect_faces(image)



