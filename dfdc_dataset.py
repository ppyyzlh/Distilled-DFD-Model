import json
import os

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import sys

from yoloface import face_analysis

face = face_analysis()


# Load YOLO model
# net = darknet.load_net("path_to_cfg_file.cfg", "path_to_weights_file.weights", 0, 1)
# meta = darknet.load_meta("path_to_data_file.data")


class DFDataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.meta_data = json.load(f)

        self.root_dir = root_dir
        self.transform = transform
        self.video_names = list(self.meta_data.keys())

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_path = os.path.join(self.root_dir, video_name)

        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # only use every 10th frame
        frames = frames[::10]
        results = []

        for frame in frames:
            # Get the bounding boxes and confidence scores for the faces in the frame
            _, boxes, confidences = face.face_detection(frame_arr=frame, model='tiny')

            # Loop through the bounding boxes
            for box in boxes:
                x, y, w, h = box
                # print(box)

                # Increase the width and height of the bounding box by 30%
                w = int(w * 1.3)
                h = int(h * 1.3)
                x -= int((w - box[2]) / 2)
                y -= int((h - box[3]) / 2)

                # Ensure that the bounding box is within the bounds of the frame
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                face_img = frame[y:y + h, x:x + w]
                # cv2.imshow("Face", face_img)
                # cv2.waitKey(1)
                results.append(face_img)
        if len(results) ==0:
            results = frames

        if self.transform:

            results = [self.transform(face_img) for face_img in results]

        label = 1 if self.meta_data[video_name]['label'] == 'REAL' else 0
        return results[0], label

