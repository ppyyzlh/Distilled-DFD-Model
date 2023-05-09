import json
import os

import cv2
import dlib
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms


class DFDataset(Dataset):
    def __init__(self, config):
        with open(config['json_file'], 'r') as f:
            self.meta_data = json.load(f)

        self.root_dir = config['root_dir']
        self.transform = self.config_transform(config['transform'])
        self.video_names = list(self.meta_data.keys())
        self.step = config['step']
        self.face_crop = config['face_crop']
        self.detector = dlib.get_frontal_face_detector()


        self.frame_counts = []  # a list to store the number of frames for each video
        self.labels = []  # a list to store the label for each video

        for video_name in self.video_names:
            video_path = os.path.join(self.root_dir, video_name)
            cap = cv2.VideoCapture(video_path)
            # get the number of frames in the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # get the label for this video
            label = 1 if self.meta_data[video_name]['label'] == 'REAL' else 0
            # append the frame count to the list
            self.frame_counts.append(frame_count)
            self.labels.append(label)  # append the label to the list

    def __len__(self):
        return sum(self.frame_counts) // self.step

    def __getitem__(self, idx):
        # find the video and frame index corresponding to the given index
        video_index = 0  # initialize the video index to 0
        frame_index = idx * self.step  # initialize the frame index to idx multiplied by x

        # loop until finding the right video
        while frame_index >= self.frame_counts[video_index]:
            # subtract the frame count of the current video from the frame index
            frame_index -= self.frame_counts[video_index]
            video_index += 1  # increment the video index by 1

        # get the video name by video index
        video_name = self.video_names[video_index]
        # get the video path by video name
        video_path = os.path.join(self.root_dir, video_name)
        label = self.labels[video_index]  # get the label by video index

        try:
            cap = cv2.VideoCapture(video_path)  # open the video file
            # set the position of the video to the frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()  # read the next frame from the video
            cap.release()  # release the video file
        except:
            ret = False  # set ret to False to indicate failure

        if ret:
            # detect faces in the frame using dlib
            if self.face_crop:
                faces = self.detector(frame)
                if len(faces) > 0:
                    # get the first face bounding box
                    face = faces[0]
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    # crop the face from the frame
                    frame = frame[y1:y2, x1:x2]
                else:
                    # create a blank frame and label
                    frame, label = self.create_blank_frame_and_label()
        else:
            # create a blank frame and label
            frame, label = self.create_blank_frame_and_label()

        # cv2.imshow("Face", frame)
        # cv2.waitKey(1)
        if self.transform:
            frame = self.transform(frame)

        to_pil = transforms.ToPILImage()
        pil_image = to_pil(frame)
        plt.imshow(pil_image)
        plt.show()

        return frame, label

    def create_blank_frame_and_label(self):
        # create a blank frame and label with zeros and -1 respectively
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        label = -1
        return frame, label
    
    def config_transform(self, config):
        transform_dict = {
            "Resize": transforms.Resize,
        }
        transform_list = [transforms.ToPILImage(), transforms.ToTensor()]

        for item in config:
            name = item['name']
            if len(item) > 1 :
                params = item.copy()
                del params['name']
                transform = transform_dict[name](**params)
            else:
                transform = transform_dict['name']
            transform_list.insert(-1, transform)
            return transforms.Compose(transform_list)
        