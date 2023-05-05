import torch
import yaml
from torch import nn, optim
from torchvision.transforms import Compose, transforms

from dfdc_dataset import DFDataset
from inception_resnet_v2 import inceptionresnetv2


def train(config):
    transform = Compose([transforms.ToPILImage(),
                         transforms.Resize((1080, 1080)),
                         transforms.CenterCrop((512, 512)),
                         transforms.ToTensor()])  # 可选的变换
    dataset = DFDataset(config['root_dir'], config['json_file'], transform)

    loader = torch.utils.data.DataLoader(
        dataset,  batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inceptionresnetv2(num_classes=2, pretrained=None)  # 初始化模型
    model.to(device)  # 将模型转移到GPU（如果有的话）
    optimizer = optim.Adam(
        model.parameters(), lr=config['learning_rate'])  # 定义优化器
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    for epoch in range(config['num_epochs']):
        for batch_idx, (frames, target) in enumerate(loader):
            data = frames.to(device)
            data, target = data.to(device), target.to(
                device)  # 将数据和标签转移到GPU（如果有的话）
            optimizer.zero_grad()  # 清空梯度
            output = model(data)  # 将数据送入模型进行前向计算
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            with torch.no_grad():
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target).sum().item()
                accuracy = correct / len(target)
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, config['num_epochs'],
                                                                                         batch_idx + 1,
                                                                                         len(loader), loss.item(
            ),
                accuracy * 100))
    torch.save(model.state_dict(), 'model_weights_teacher.pt')


def test(config):
    transform = Compose([transforms.ToPILImage(),
                         transforms.Resize((1080, 1080)),
                         transforms.CenterCrop((512, 512)),
                         transforms.ToTensor()])  # 可选的变换
    dataset = DFDataset(config['root_dir'], config['json_file'], transform)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inceptionresnetv2(num_classes=2, pretrained=None)
    model.load_state_dict(torch.load(
        config['model_weights'], map_location=device))
    model.to(device)
    model.eval()

    total_correct = 0
    with torch.no_grad():
        for batch_idx, (frames, target) in enumerate(loader):
            data = frames.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            total_correct += correct

    accuracy = total_correct / len(dataset)
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    config = dict()
    with open('train_teacher.yaml') as f:
        config = yaml.safe_load(f)
    if config['test']:
        test(config)
    else:
        train(config)
