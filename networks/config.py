from .inceptionresnetv2 import inceptionresnetv2
from efficientnet_pytorch import EfficientNet
from torch import nn

def config_model(model_name, num_classes):
    if model_name == "inceptionresnetv2":
            model = inceptionresnetv2(num_classes) 
    elif model_name == "efficientnet-b7":
            model = EfficientNet.from_name('efficientnet-b7')
            model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model
            