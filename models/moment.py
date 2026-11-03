"""MOMENT (CMU, slide 10), frozen: each window -> embedding, then Ridge regression.

MOMENT reads every channel separately, always 512 frames long: our windows are padded on the
left and a mask tells the model which frames are real. Its official embed() gives one vector per
patch of 8 frames: we average the real patches and put the channels side by side.
"""
import numpy as np
import torch
from momentfm import MOMENTPipeline

import config
from pipeline.evaluation import fit_ridge
from pipeline.torch_training import DEVICE

INPUT = "embeddings"
LENGTH, PATCH = 512, 8  # fixed by MOMENT


def embed(windows):
    model = MOMENTPipeline.from_pretrained(config.FOUNDATION_CHECKPOINTS["moment"], model_kwargs={"task_name": "embedding"})
    model.init()  # its warning "only reconstruction head is pre-trained" does not matter: we use no head
    model = model.to(DEVICE).eval()
    n, channels, frames = windows.shape
    padded = np.zeros((n, channels, LENGTH), np.float32)
    padded[:, :, -frames:] = windows
    mask = np.zeros((n, LENGTH), np.float32)
    mask[:, -frames:] = 1  # 1 = real frame, 0 = padding
    real_patches = frames // PATCH
    out = []
    with torch.no_grad():
        for i in range(0, n, 16):
            result = model.embed(x_enc=torch.tensor(padded[i:i + 16], device=DEVICE),
                                 input_mask=torch.tensor(mask[i:i + 16], device=DEVICE), reduction="none")
            patches = result.embeddings[:, :, -real_patches:]  # windows x channels x patches x size
            out.append(patches.mean(dim=2).reshape(len(patches), -1).float().cpu().numpy())
    return np.concatenate(out)


def fit(X, y, groups):
    return fit_ridge(X, y, groups)


def predict(model, X):
    return model.predict(X)
