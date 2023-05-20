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
    model = config_model(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])  # 定义优化器
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)

    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_model(model, loader, optimizer, criterion, accuracy, device) # 训练模型并获取训练损失和准确率
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc.item() * 100:.2f}%")
        val_loss, val_acc = evaluate_model(model, loader, criterion, accuracy, device) # 评估模型并获取验证损失和准确率
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc.item() * 100:.2f}%")
        torch.save(model.state_dict(), os.path.join(config['trained_model_dir'], f'model_weights_{epoch+1}_{round(train_acc.item() * 100, 2)}%.pt'))


def test(config):
    dataset = DFDataset(config['test_dataset']) # 加载测试集

    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']) # 创建数据加载器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 获取设备
    model = config_model(device) # 创建模型
    model.load_state_dict(torch.load(config['test_model_weight'])) # 加载最佳模型的权重

    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device) # 定义准确率指标

    test_loss, test_acc = evaluate_model(model, loader, criterion, accuracy, device) # 评估模型并获取测试损失和准确率
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc.item() * 100:.2f}%') # 打印测试结果



def config_model(device):
    model = EfficientNet.from_name('efficientnet-b7')
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.to(device)  # 将模型转移到GPU（如果有的话）
    return model

    
if __name__ == '__main__':
    config = dict()
    with open('train_student.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config['test']:
        test(config)
    else:
        train(config)
