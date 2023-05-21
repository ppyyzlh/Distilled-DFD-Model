from .ModelWrapper import ModelWrapper

import torch
from models.config import config_model

class DistillationModelWrapper(ModelWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.teacher_model = config_model(config['teacher_model'], config['num_classes'])
        self.teacher_model.load_state_dict(torch.load(config['teacher_model_weight']))