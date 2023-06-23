
import torch
import torchmetrics
from tqdm import tqdm
from torch import nn

from dataloader.dfdc_loader import DFDCLoader
from dataloader.video_dataset import VideoDataset


from .ModelWrapper import ModelWrapper


class TestModelWrapper(ModelWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.model.load_state_dict(torch.load(config['test_model_weight']))
        # create a test dataset
        loader = DFDCLoader(config['loader'])
        dataset =VideoDataset(config['dataset'], loader)
        # create a test loader
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        # create an accuracy metric
        self.accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=config['num_classes'])
        
        criterion_name = config['criterion']['name']
        criterion_params = config['criterion'].get('params', {})
        criterion_class = getattr(nn, criterion_name) # 获取损失函数类
        self.criterion = criterion_class(**criterion_params) # 创建损失函数实例


    def test(self):
        # test the model on the test dataset
        self.accuracy.reset()
        sum_loss = 0
        loop = tqdm(self.loader) # create a loop object
        self.accuracy.to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad(): # do not compute gradients
            for batch_idx, (frames, labels) in enumerate(self.loader):
                frames, labels= frames.to(self.device), labels.to(self.device)  # move data and labels to GPU (if any)
                output = self.model(frames)  # feed data into the model for forward computation
                loss = self.criterion(output, labels)  # compute loss

                sum_loss += loss.item()
                self.accuracy.update(output, labels)
                loop.set_postfix(loss=round(sum_loss/(batch_idx+1), 2), acc=f'{self.accuracy.compute().item() * 100:.2f}%') # update the progress bar information
                loop.update()
        
        loop.close()
        avg_loss = sum_loss / len(self.loader)
        acc = self.accuracy.compute()
        return avg_loss, acc