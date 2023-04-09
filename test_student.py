import torch
import torch.nn as nn
from torchvision.transforms import Compose, transforms
from dfdc_dataset import DFDataset
from inception_resnet_v2 import inceptionresnetv2

def test():
    transform = Compose([transforms.ToPILImage(),
                         transforms.Resize((1080, 1080)),
                         transforms.CenterCrop((512, 512)),
                         transforms.ToTensor()])
    root_dir = '/home/cuc/Public_Data_Set/dfdc/dfdc_train_part_1'
    json_file = '/home/cuc/Public_Data_Set/dfdc/dfdc_train_part_1/metadata.json'
    dataset = DFDataset(root_dir, json_file, transform)

    batch_size = 8
    shuffle = False
    num_workers = 0
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inceptionresnetv2(num_classes=2, pretrained=None)
    # model = EfficientNet.from_name('efficientnet-b7')
    # model._fc = nn.Linear(model._fc.in_features, 2)
    model.last_linear = nn.Linear(model.last_linear.in_features, 2) # 替换为新的全连接层
    model.to(device)
    model.load_state_dict(torch.load('model_weights.pt'))
    model.eval()

    with torch.no_grad():
        total_correct = 0
        for batch_idx, (frames, target) in enumerate(test_loader):
            data, target = frames.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            total_correct += correct

        accuracy = total_correct / len(dataset)
        print('Accuracy: {:.2f}%'.format(accuracy * 100))

if __name__ == '__main__':
    test()

