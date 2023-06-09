from torch.utils.data import Dataset
import json
from .video import Video
import os               
from tqdm import tqdm




class VideoDataset(Dataset):
    def __init__(self, config, loader):
        self.loader = loader
        self.videos = []
        for item in config:
            self.load_video(item['root_dir'], item['metadata'])
        self.total_frames = 0
        for video in self.videos:
            self.total_frames += video.frame_count

    def load_video(self, root_dir, meta_data):
        print(f'load {meta_data}')
        with open(meta_data, 'r') as f:
            meta_data = json.load(f)
        for video_name in  tqdm(meta_data.keys()):
            if os.access(os.path.join(root_dir, video_name), os.R_OK) :
                attributes = meta_data[video_name]
                self.videos.append(Video(root_dir, video_name, attributes))
            else:
                tqdm.write(f"Cannot access {video_name} in {root_dir}")
        return

    def __len__(self):
        return self.total_frames // self.loader.step

    def __getitem__(self, index):
        video_index = 0
        frame_index = index * self.loader.step
        while frame_index >= self.videos[video_index].frame_count:
            frame_index -= self.videos[video_index].frame_count
            video_index += 1
        video = self.videos[video_index]
        try:
            frame = video.get_frame(frame_index)
            label = video.label
            if self.loader.face_crop:
                    frame = self.loader.crop_face(frame)
            if frame is None:
                frame, label = self.loader.create_blank_frame_and_label()
            frame = self.loader.transform(frame)
        except Exception as e:
            print(f'Error in {video.name} {frame_index}')
            print(e)
            frame, label = self.loader.create_blank_frame_and_label()
            frame = self.loader.transform(frame)
        return (frame, label)
