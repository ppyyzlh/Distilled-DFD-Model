from dfdc_dataset import DFDataset
import yaml
import cv2
import torch
from torchvision.transforms import Compose, transforms


config = dict()
with open('train_student.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)
dataset_config = dict()
with open('dateset.yaml', encoding='utf-8') as f:
    dataset_config = yaml.safe_load(f)

transform = Compose([transforms.ToPILImage(),
                        transforms.Resize((64, 64)),
                        transforms.ToTensor()])  # 可选的变换


dataset = DFDataset(dataset_config, transform)

loader = torch.utils.data.DataLoader(
        dataset,  batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])
for batch_idx, (frames, target) in enumerate(loader):
    # cv2.imshow("Face", frames[0])
    print(target)
    print(batch_idx)
    pass