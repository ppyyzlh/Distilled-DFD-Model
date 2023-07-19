import dlib
import numpy as np
from torchvision import transforms


class DFDCPreprocessor:
    def __init__(self, config) -> None:
        self.transform = self.config_transform(config['transform'])
        self.step = config['step']
        self.face_crop = config['face_crop']
        self.detector = dlib.get_frontal_face_detector()

    def config_transform(self, config):
        transform_dict = {
            "Resize": transforms.Resize,
            "CenterCrop": transforms.CenterCrop,
        }
        transform_list = [transforms.ToPILImage()]

        for item in config:
            name = item.pop('name')
            transform = transform_dict[name]
            if transform is not None:
                transform = transform(**item)
                transform_list.append(transform)
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    def crop_face(self, frame):
        faces = self.detector(frame)
        if len(faces) > 0:
            # get the first face bounding box
            face = faces[0]
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            # crop the face from the frame
            return frame[y1:y2, x1:x2]
        else:
            return None

    def create_blank_frame_and_label(self):
        # create a blank frame and label with zeros and -1 respectively
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        label = 1
        return frame, label
