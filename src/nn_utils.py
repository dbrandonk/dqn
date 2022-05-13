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

def train(epoch, data_loader, model, optimizer, criterion):

    acc = MeanMeter()
    iter_time = MeanMeter()
    losses = MeanMeter()

    for idx, (data, target) in enumerate(data_loader):

        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        model.train()
        out = model(data)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calc accuracy
        value, preds = out.max(dim=-1)
        batch_acc = preds.eq(target).sum() / target.shape[0]

        # Calc means
        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])
        iter_time.update(time.time() - start)

    return losses.mean.detach().cpu().numpy(), acc.mean.detach().cpu().numpy()

def predict(model):

    model.eval()
    with torch.no_grad():
      out = model(data)

  return out

