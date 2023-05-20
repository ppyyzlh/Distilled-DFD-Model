import torch
from tqdm import tqdm

def train_model(model, loader, optimizer, criterion, accuracy, device):
    # 训练模型的函数
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
    avg_loss = sum_loss / len(loader)
    acc = accuracy.compute()
    return avg_loss, acc

def evaluate_model(model, loader, criterion, accuracy, device):
    # 评估模型的函数
    accuracy.reset()
    sum_loss = 0
    loop = tqdm(loader) # 创建一个循环对象
    with torch.no_grad(): # 不计算梯度
        for batch_idx, (frames, labels) in enumerate(loader):
            frames, labels= frames.to(device), labels.to(device)  # 将数据和标签转移到GPU（如果有的话）
            output = model(frames)  # 将数据送入模型进行前向计算
            loss = criterion(output, labels)  # 计算损失

            sum_loss += loss.item()
            accuracy.update(output, labels)
            loop.set_postfix(loss=round(sum_loss/(batch_idx+1), 2), acc=f'{accuracy.compute().item() * 100:.2f}%') # 更新进度条的信息
            loop.update()
    
    loop.close()
    avg_loss = sum_loss / len(loader)
    acc = accuracy.compute()
    return avg_loss, acc