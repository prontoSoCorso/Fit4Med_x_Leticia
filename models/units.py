"""UniTS (slide 12): one transformer built for many time-series tasks, trained from scratch on the windows.

It uses the original model code (models/third_party/units_original.py, MIT licence, unchanged)
through its 'classification' path with a single output, trained with the squared error: a regressor.
Starting from pretrained UniTS weights would be fine-tuning: later, not now. Fixed settings, early stopping.
"""
from types import SimpleNamespace

import torch.nn as nn

from models.third_party.units_original import Model
from pipeline import torch_training

INPUT = "windows"
SETTINGS = SimpleNamespace(d_model=64, n_heads=4, e_layers=3, patch_len=16, stride=16, prompt_num=10, dropout=0.1)


class UniTSRegressor(nn.Module):  # PyTorch needs a class for networks with custom steps
    def __init__(self, n_channels):
        super().__init__()
        task = [("mbi", {"dataset": "fit4med", "task_name": "classification", "enc_in": n_channels, "num_class": 1})]
        self.units = Model(SETTINGS, task)

    def forward(self, x):  # x: windows x channels x frames; UniTS wants windows x frames x channels
        return self.units(x.transpose(1, 2), None, task_id=0, task_name="classification")


def fit(X, y, groups):
    return torch_training.train(UniTSRegressor(X.shape[1]), X, y, groups)


def predict(model, X):
    return torch_training.predict(model, X)
