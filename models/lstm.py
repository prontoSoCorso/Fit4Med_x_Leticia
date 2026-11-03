"""LSTM (long short-term memory, slide 8), trained end to end on the windows.

It reads a window step by step and keeps a memory of what it has seen; its final memory is
turned into the label. Windows are read at 30 frames per second (every second frame), because
long sequences make LSTMs slow and hard to train. Fixed settings with early stopping.
"""
import torch.nn as nn

from pipeline import torch_training

INPUT = "windows"


class LSTMRegressor(nn.Module):  # PyTorch needs a class for networks with custom steps
    def __init__(self, n_channels, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: windows x channels x frames
        x = x[:, :, ::2].transpose(1, 2)  # every second frame, as windows x frames x channels
        _, (memory, _) = self.lstm(x)
        return self.head(memory[-1])


def fit(X, y, groups):
    return torch_training.train(LSTMRegressor(X.shape[1]), X, y, groups)


def predict(model, X):
    return torch_training.predict(model, X)
