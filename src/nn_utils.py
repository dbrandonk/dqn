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

        if idx % 100 == 0:
          print(('Epoch: [{0}][{1}/{2}]\t'
                  'Time {iter_time.val:.3f} ({iter_time.mean:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.mean:.4f})\t'
                  'Accuracy {accu.val:.4f} ({accu.mean:.4f})\t')
                .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, accu=acc))

    return losses.mean.detach().cpu().numpy(), acc.mean.detach().cpu().numpy()

def validate(epoch, val_loader, model, criterion):

  acc = MeanMeter()
  iter_time = MeanMeter()
  losses = MeanMeter()

  # evaluation loop
  for idx, (data, target) in enumerate(val_loader):
    start = time.time()

    if torch.cuda.is_available():
      data = data.cuda()
      target = target.cuda()

    model.eval()
    with torch.no_grad():
      out = model(data)
      loss = criterion(out, target)

    value, preds = out.max(dim=-1)
    batch_acc = preds.eq(target).sum() / target.shape[0]

    acc.update(batch_acc, out.shape[0])
    iter_time.update(time.time() - start)
    losses.update(loss, out.shape[0])


  print("")
  print("Validation Metrics:")


  print(("Accuracy mean total: {accuracy.mean:.4f}").format(accuracy=acc))
  print(("Loss mean: {loss.mean:.4f}").format(loss=losses))
  print("")

  return losses.mean.detach().cpu().numpy(), acc.mean.detach().cpu().numpy()

