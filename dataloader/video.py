import os

import cv2


class Video:
    def __init__(self, root_dir, name, attributes):
        self.name = name
        self.path = os.path.join(root_dir, name)
        self.label = 1 if attributes['label'] == 'REAL' else 0
        self.original = attributes['original']
        self.original = attributes['split']
        self.cap = cv2.VideoCapture(self.path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.release()

    def get_frame(self, index):
        if index >= 0 and index < self.frame_count:
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
            else:
                return None
        else:
            raise ValueError("Invalid frame index")
