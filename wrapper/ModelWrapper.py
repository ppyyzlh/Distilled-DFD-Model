
import torch
from networks.config import config_model


class ModelWrapper:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = config_model(config['model'], config['num_classes'])