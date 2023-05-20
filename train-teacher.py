from models.inceptionresnetv2 import inceptionresnetv2

import os

import torch
import torchmetrics
import yaml
from efficientnet_pytorch import EfficientNet
from torch import nn, optim

from dfdc_dataset import DFDataset
from model_utils import evaluate_model, train_model


def train(config):

    if not os.path.exists(config['trained_model_dir']):
        os.makedirs(config['trained_model_dir'])

    dataset = DFDataset(config['dataset'])

    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inceptionresnetv2(num_classes=2)  # 初始化模型
    model.to(device)  # 将模型转移到GPU（如果有的话）

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])  # 定义优化器
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)

    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_model(model, loader, optimizer, criterion, accuracy, device) # 训练模型并获取训练损失和准确率
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc.item() * 100:.2f}%")
        val_loss, val_acc = evaluate_model(model, loader, criterion, accuracy, device) # 评估模型并获取验证损失和准确率
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc.item() * 100:.2f}%")
        torch.save(model.state_dict(), os.path.join(config['trained_model_dir'], f'model_weights_{epoch+1}_{round(train_acc.item() * 100, 2)}%.pt'))

if __name__ == '__main__':
    config = dict()
    with open('train_teacher.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train(config)
