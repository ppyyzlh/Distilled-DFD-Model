import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn, optim
from torchvision.transforms import Compose, transforms

from dfdc_dataset import DFDataset
from inception_resnet_v2 import inceptionresnetv2

import yaml


def train_distillation(config):
    """使用蒸馏训练方法训练学生模型

    Args:
        root_dir: str, 包含视频文件的根目录
        json_file: str, 包含元数据的JSON文件的路径
        model_weights_teacher: str, 学生模型的预训练权重文件的路径
        model_weights_student: str, 教师模型的预训练权重文件的路径
        batch_size: int, 每个训练批次的样本数 (default: 1)
        shuffle: bool, 是否在每个epoch之前打乱数据集 (default: True)
        num_workers: int, 用于数据加载的工作线程数 (default: 0)
        learning_rate: float, Adam优化器的学习率 (default: 0.001)
        num_epochs: int, 训练的epoch数 (default: 10)
        temperature: float, 温度参数 (default: 10.0)
        alpha: float, 用于加权损失的参数alpha (default: 0.5)

    Returns:
        None
    """
    transform = Compose([transforms.ToPILImage(),
                         transforms.Resize((1080, 1080)),
                         transforms.CenterCrop((512, 512)),
                         transforms.ToTensor()])  # 可选的变换

    dataset = DFDataset(config['root_dir'], config['json_file'], transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'])

    # 定义教师模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = inceptionresnetv2(num_classes=2, pretrained=None)
    teacher_model.load_state_dict(torch.load(config['model_weights_teacher'], map_location=device))
    teacher_model.to(device)
    teacher_model.eval()  # 将教师模型设置为评估模式

    # model = inceptionresnetv2(num_classes=2, pretrained=None)  # 初始化模型
    student_model = EfficientNet.from_name('efficientnet-b7')
    student_model._fc = nn.Linear(student_model._fc.in_features, 2)
    ##是否加载预训练学生模型
    student_model.load_state_dict(torch.load(config['model_weights_student'], map_location=device))
    student_model.to(device)  # 将模型转移到GPU（如果有的话）

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student_model.parameters(), lr=0.001) 

    # 定义温度和权重
    temperature = config['temperature']
    alpha = config['alpha']

    # 训练学生模型
    for epoch in range(config['num_epochs']):
        for batch_idx, (frames, target) in enumerate(loader):
            data = frames.to(device)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  

            with torch.no_grad():
                teacher_output = teacher_model(data) / temperature  # 计算教师模型的输出
            student_output = student_model(data) / temperature  # 计算学生模型的输出

            # 计算蒸馏损失
            loss = alpha * criterion(F.log_softmax(student_output.float(), dim=1),
                                 F.softmax(teacher_output.float(), dim=1)) * temperature * temperature + \
               (1 - alpha) * criterion(student_output.float(), target.float())

            loss.backward()  
            optimizer.step()  

            # 计算准确率
            with torch.no_grad():
                _, predicted = torch.max(student_output.data, 1)
                correct = (predicted == target).sum().item()
                accuracy = correct / len(target)
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, 10,
                                                                                         batch_idx + 1,
                                                                                         len(loader), loss.item(),
                                                                                         accuracy * 100))
    # 保存学生模型权重
    torch.save(student_model.state_dict(), 'model_weights_student_distill.pt')

    
    
if __name__ == '__main__':
    config = dict()
    with open('train_distillation.yaml') as f:
        config = yaml.safe_load(f)
    train_distillation(config)