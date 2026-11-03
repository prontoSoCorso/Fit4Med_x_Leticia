"""1D convolutional neural network (1D-CNN, slide 8), trained end to end on the windows.

Small filters slide along time and learn local movement shapes; pooling then summarises the
whole window into a few numbers, which a last layer turns into the label.
Fixed settings (config.DEEP) with early stopping: too few patients to tune a network.
"""
import torch.nn as nn

from pipeline import torch_training

INPUT = "windows"


def make_net(n_channels):
    return nn.Sequential(
        nn.Conv1d(n_channels, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))


def fit(X, y, groups):
    return torch_training.train(make_net(X.shape[1]), X, y, groups)


def predict(model, X):
    return torch_training.predict(model, X)
