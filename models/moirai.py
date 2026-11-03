"""MOIRAI-2 (Salesforce, slide 10), frozen: each window -> embedding, then Ridge regression.

MOIRAI is built for forecasting and has no official embedding function: we record the output of
its transformer encoder (with a 'hook') while it makes a short forecast that we then ignore.
The encoder gives one vector per patch of each channel: we average the patches of the window and
put the channels side by side. The weights' licence (CC-BY-NC 4.0) allows research use only.
"""
import numpy as np
import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

import config
from pipeline.evaluation import fit_ridge
from pipeline.torch_training import DEVICE

INPUT = "embeddings"


def embed(windows):
    n, channels, frames = windows.shape
    module = Moirai2Module.from_pretrained(config.FOUNDATION_CHECKPOINTS["moirai"])
    forecaster = Moirai2Forecast(module=module, prediction_length=module.patch_size, context_length=frames,
                                 target_dim=channels, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    forecaster = forecaster.to(DEVICE).eval()
    patches = -(-frames // module.patch_size)  # patches per channel (rounded up)
    recorded = []
    hook = module.encoder.register_forward_hook(lambda layer, inputs, output: recorded.append(output.cpu()))
    with torch.no_grad():
        for i in range(0, n, 32):
            x = torch.tensor(windows[i:i + 32], device=DEVICE).transpose(1, 2)  # windows x frames x channels
            forecaster(past_target=x, past_observed_target=torch.ones_like(x, dtype=torch.bool),
                       past_is_pad=torch.zeros(x.shape[:2], dtype=torch.bool, device=DEVICE))
    hook.remove()
    # The encoder output starts with the window's patches, channel after channel, then the forecast patches
    window_tokens = torch.cat([r[:, :channels * patches] for r in recorded])
    return window_tokens.reshape(n, channels, patches, -1).mean(dim=2).reshape(n, -1).float().numpy()


def fit(X, y, groups):
    return fit_ridge(X, y, groups)


def predict(model, X):
    return model.predict(X)
