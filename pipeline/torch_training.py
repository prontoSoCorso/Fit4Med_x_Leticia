"""Training loop shared by the neural networks (1D-CNN, LSTM, PatchTST, UniTS).

A neural network learns in small steps: it predicts a batch of windows, measures its error (loss)
and adjusts its weights; one pass over all training windows is an epoch. To stop before it
memorises the training patients (overfitting), some training patients are kept aside for
validation and training stops when their error no longer improves (early stopping).
The test patient is never used here.
"""
import copy

import numpy as np
import torch

import config
from pipeline.evaluation import split_validation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # a GPU makes training much faster


def outputs(net, X):
    """Network outputs for many windows, in batches, without training."""
    net.eval()
    with torch.no_grad():
        return torch.cat([net(X[i:i + 256].to(DEVICE)).reshape(-1).cpu() for i in range(0, len(X), 256)]).numpy()


def train(net, X, y, groups):
    """Train `net` on windows X (windows x channels x frames); return what predict() needs."""
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    # Standardise each channel and the label with numbers from the training windows only
    mean, std = X.mean(axis=(0, 2), keepdims=True), X.std(axis=(0, 2), keepdims=True) + 1e-8
    y_mean, y_std = y.mean(), y.std() + 1e-8
    Xn = torch.tensor((X - mean) / std, dtype=torch.float32)
    yn = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    fit_rows, val_rows = split_validation(groups)

    settings = config.DEEP
    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings["learning_rate"])
    best_weights, best_loss, epochs_without_progress = copy.deepcopy(net.state_dict()), np.inf, 0
    for epoch in range(settings["max_epochs"]):
        net.train()
        order = rng.permutation(fit_rows)
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            if len(rows) < 2:
                continue  # some layers need at least 2 windows in a batch
            loss = torch.nn.functional.mse_loss(net(Xn[rows].to(DEVICE)).reshape(-1), yn[rows].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss = np.mean((outputs(net, Xn[val_rows]) - yn[val_rows].numpy()) ** 2)
        if val_loss < best_loss:
            best_weights, best_loss, epochs_without_progress = copy.deepcopy(net.state_dict()), val_loss, 0
        else:
            epochs_without_progress += 1
            if epochs_without_progress >= settings["patience"]:
                break
    net.load_state_dict(best_weights)  # keep the epoch with the lowest validation error
    return {"net": net, "mean": mean, "std": std, "y_mean": y_mean, "y_std": y_std}


def predict(model, X):
    Xn = torch.tensor((X - model["mean"]) / model["std"], dtype=torch.float32)
    return outputs(model["net"], Xn) * model["y_std"] + model["y_mean"]
