

class BaseDetector:
    def __init__(self, mode='RGB'):
        self.mode = ''

    def detect_faces(self, image):
        raise NotImplementedError(f"{type(self).__name__}.detect_faces was not implemented")

