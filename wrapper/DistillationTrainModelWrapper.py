import os

import torch
from torch import nn, optim
import torchmetrics
from tqdm import tqdm

from dataloader.dfdc_loader import DFDCLoader
from dataloader.video_dataset import VideoDataset

import torch.nn.functional as F

from .DistillationModelWrapper import DistillationModelWrapper


class DistillationTrainModelWrapper(DistillationModelWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.trained_model_dir = config['trained_model_dir']
        if not os.path.exists(config['trained_model_dir']):
            os.makedirs(config['trained_model_dir'])
        
        loader = DFDCLoader(config['loader'])
        dataset =VideoDataset(config['dataset'], loader)
        val_dataset =VideoDataset(config['val_dataset'], loader)
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])

        self.accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=config['num_classes'])
        self.num_epochs = config['num_epochs']


        optimizer_name = config['optimizer']['name']
        optimizer_params = config['optimizer'].get('params', {})
        optimizer_class = getattr(optim, optimizer_name) # 获取优化器类
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params) # 创建优化器实例

        criterion_name = config['criterion']['name']
        criterion_params = config['criterion'].get('params', {})
        criterion_class = getattr(nn, criterion_name) # 获取损失函数类
        self.criterion = criterion_class(**criterion_params) # 创建损失函数实例

        temperature = config['temperature']
        alpha = config['alpha']
        # 计算学生输出和真实标签之间的交叉熵，并乘以权重的补数

        # self.distill_loss = lambda output, teacher_output, labels: (
        #         F.kl_div(F.log_softmax(output / temperature, dim=1),
        #         teacher_output / temperature,
        #         log_target=True,
        #         reduction='batchmean') * (temperature ** 2) +
        #         (1 - alpha) * F.nll_loss(output, labels)
        # )

        self.distill_loss = lambda output, teacher_output, labels: (
                F.kl_div(F.log_softmax(output / temperature, dim=1),
                F.softmax(teacher_output / temperature, dim=1),
             reduction='batchmean') * alpha +
    # 计算学生输出和真实标签之间的交叉熵，使用log_softmax函数和nll_loss函数，并乘以权重的补数
    (1 - alpha) * nn.CrossEntropyLoss()(output, labels)
)
 
    def train(self):
        # 训练模型的函数
        self.accuracy.reset()
        loop = tqdm(self.loader) # 创建一个循环对象
        sum_loss = 0
        self.accuracy.to(self.device)
        self.model.to(self.device)
        self.model.train()
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for batch_idx, (frames, labels) in enumerate(self.loader):
            frames, labels= frames.to(self.device), labels.to(self.device)  # 将数据和标签转移到GPU（如果有的话）
            with torch.no_grad():
                    teacher_output = self.teacher_model(frames)
            output = self.model(frames)  # 将数据送入模型进行前向计算
            loss = self.distill_loss(output, teacher_output, labels)  # 计算损失
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新权重

            sum_loss += loss.item()
            self.accuracy.update(output, labels)
            loop.write(str(loss.item()))
            loop.set_postfix(loss=round(sum_loss/(batch_idx+1), 2), acc=f'{self.accuracy.compute().item() * 100:.2f}%') # 更新进度条的信息
            loop.update()
            
        loop.close()
        avg_loss = sum_loss / len(self.loader)
        acc = self.accuracy.compute()
        return avg_loss, acc

    def evaluate(self):
        # 评估模型的函数
        self.accuracy.reset()
        sum_loss = 0
        loop = tqdm(self.loader) # 创建一个循环对象
        self.accuracy.to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad(): # 不计算梯度
            for batch_idx, (frames, labels) in enumerate(self.loader):
                frames, labels= frames.to(self.device), labels.to(self.device)  # 将数据和标签转移到GPU（如果有的话）
                output = self.model(frames)  # 将数据送入模型进行前向计算
                loss = self.criterion(output, labels)  # 计算损失

                sum_loss += loss.item()
                self.accuracy.update(output, labels)
                loop.set_postfix(loss=round(sum_loss/(batch_idx+1), 2), acc=f'{self.accuracy.compute().item() * 100:.2f}%') # 更新进度条的信息
                loop.update()
        
        loop.close()
        avg_loss = sum_loss / len(self.loader)
        acc = self.accuracy.compute()
        return avg_loss, acc
