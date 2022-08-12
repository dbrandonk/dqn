import torch


def train(data, target, model, optimizer, criterion):

    data = torch.from_numpy(data)
    target = torch.from_numpy(target)

    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    model.train()
    out = model(data)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


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
