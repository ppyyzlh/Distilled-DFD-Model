from .inceptionresnetv2 import inceptionresnetv2
from efficientnet_pytorch import EfficientNet
from torch import nn

def config_model(model_name, num_classes):
    match model_name:
        case "inceptionresnetv2":
            model = inceptionresnetv2(num_classes) 
        case "efficientnet-b7":
            model = EfficientNet.from_name('efficientnet-b7')
            model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model
            