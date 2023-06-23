import os
import argparse

import torch
import yaml
from wrapper.TrainModelWrapper import TrainModelWrapper

def train(wrapper):
    for epoch in range(wrapper.num_epochs):
        train_loss, train_acc = wrapper.train()
        print(f"Epoch [{epoch + 1}/{wrapper.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc.item() * 100:.2f}%")
        val_loss, val_acc = wrapper.evaluate()
        print(f"Epoch [{epoch + 1}/{wrapper.num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc.item() * 100:.2f}%")
        torch.save(wrapper.model.state_dict(), os.path.join(wrapper.trained_model_dir, f'model_weights_{epoch+1}_{round(train_acc.item() * 100, 2)}%.pt'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dev/train_student.yaml', help='path to the config file')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    wrapper = TrainModelWrapper(config)
    train(wrapper)