import time
import torch


class MeanMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count


def train(epoch, data, target, model, optimizer, criterion):

    data = torch.from_numpy(data)
    target = torch.from_numpy(target)

    acc = MeanMeter()
    losses = MeanMeter()

    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    model.train()
    out = model(data)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return


def predict(model, data):

    data = torch.from_numpy(data)
    if torch.cuda.is_available():
        model = model.cuda()
        data = data.cuda()
    model.eval()
    with torch.no_grad():
        out = model(data)

    out = out.cpu().detach().numpy()

    return out
