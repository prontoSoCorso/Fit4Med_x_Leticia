"""PatchTST (slide 12), from Hugging Face transformers, trained from scratch on the windows.

Each channel is cut into short pieces (patches, like words in a sentence) and a transformer
reads them; every channel is handled separately with shared weights (channel independence).
Fixed settings with early stopping; no pretraining (it would need many more recordings).
"""
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForRegression

from pipeline import torch_training

INPUT = "windows"


class PatchTSTRegressor(nn.Module):  # PyTorch needs a class for networks with custom steps
    def __init__(self, n_channels, n_frames):
        super().__init__()
        settings = PatchTSTConfig(num_input_channels=n_channels, context_length=n_frames, num_targets=1,
                                  patch_length=16, patch_stride=8, d_model=64, num_attention_heads=4,
                                  num_hidden_layers=3, ffn_dim=128, dropout=0.2, head_dropout=0.2)
        self.patchtst = PatchTSTForRegression(settings)

    def forward(self, x):  # x: windows x channels x frames; PatchTST wants windows x frames x channels
        return self.patchtst(past_values=x.transpose(1, 2)).regression_outputs


def fit(X, y, groups):
    return torch_training.train(PatchTSTRegressor(X.shape[1], X.shape[2]), X, y, groups)


def predict(model, X):
    return torch_training.predict(model, X)
