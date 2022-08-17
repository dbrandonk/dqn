import torch
import numpy as np


def train(data, target, model, optimizer, criterion, device):

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    data = data.to(device)
    target = target.to(device)

    model.train()
    out = model(data)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def predict(model, data, device):

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    data = data.to(device)

    model.eval()
    with torch.no_grad():
        out = model(data)

    out = out.cpu().detach().numpy()

    return out
