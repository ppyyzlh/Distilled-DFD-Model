import torch
import yaml
import torchmetrics
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from torch import nn, optim
from torchvision.transforms import Compose, transforms

from dfdc_dataset import DFDataset


def train(config):

    dataset = DFDataset(config['dataset'])

    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config_model(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])  # 定义优化器
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)

    for epoch in range(config['num_epochs']):
        accuracy.reset()
        sum_loss = 0
        loop = tqdm(loader) # 创建一个循环对象
        for batch_idx, (frames, labels) in enumerate(loader):
            frames, labels= frames.to(device), labels.to(device)  # 将数据和标签转移到GPU（如果有的话）
            optimizer.zero_grad()  # 清空梯度
            output = model(frames)  # 将数据送入模型进行前向计算
            loss = criterion(output, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            sum_loss += loss.item()
            accuracy.update(output, labels)
            loop.set_postfix(loss=round(sum_loss/(batch_idx+1), 2), acc=f'{accuracy.compute().item() * 100:.2f}%') # 更新进度条的信息
            loop.update()
            
        loop.close()
        avg_loss = sum_loss // len(loader)
        acc = accuracy.compute()
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, config['num_epochs'], avg_loss, acc * 100))

    torch.save(model.state_dict(), 'model_weights_student.pt')


def test(config):
    transform = Compose([transforms.ToPILImage(),
                         transforms.Resize((1080, 1080)),
                         transforms.CenterCrop((512, 512)),
                         transforms.ToTensor()])  # 可选的变换
    dataset = DFDataset(config['root_dir'], config['json_file'], transform)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_name('efficientnet-b7')
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(torch.load(
        config['model_weights'], map_location=device))
    model.to(device)  # 将模型转移到GPU（如果有的话）

    total_correct = 0  # 总正确数
    with torch.no_grad():
        for batch_idx, (frames, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)  # 将数据和标签转移到GPU（如果有的话）
            output = model(data)  # 将数据送入模型进行前向计算

            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            total_correct += correct

    accuracy = total_correct / len(dataset)  # 计算准确率
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))


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
