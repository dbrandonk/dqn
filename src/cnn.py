import copy
import time
import torch
import torch.nn as nn

try:
    from data import import_data
    from nn_utils import train
    from nn_utils import validate
except:
    print('failed to import some modules')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        outs = nn.functional.relu(self.conv1(x))
        outs = self.pool(nn.functional.relu(self.conv2(outs)))
        outs = nn.functional.relu(self.conv3(outs))
        outs = self.pool(nn.functional.relu(self.conv4(outs)))
        outs = nn.functional.relu(self.conv5(outs))
        outs = self.pool(nn.functional.relu(self.conv6(outs)))
        outs = torch.flatten(outs, 1)
        outs = nn.functional.relu(self.fc1(outs))
        outs = nn.functional.relu(self.fc2(outs))
        outs = self.fc3(outs)
        return outs

def cnn_run(batch_size=None, epochs=None, learning_rate=None, weight_decay=None, save_path='..'):

    dataloader_train, dataloader_val, dataloader_test = import_data(batch_size)

    model = CNN()
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_losses = []
    train_accs = []

    val_losses = []
    val_accs = []

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):

        # train loop
        train_loss, train_acc  = train(epoch, dataloader_train, model, optimizer, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # validation loop
        val_loss, val_acc = validate(epoch, dataloader_val, model, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)


    print('Best Acc: {:.4f}'.format(best_acc))
    print('')

    run_time = time.time() - start_time

    test_loss, test_acc = validate(epoch, dataloader_test, model, criterion)
    test_results = open(save_path + '/output/cnn_model_tr.txt', 'w')
    test_results.write('test loss: ' + str(test_loss) + '\n')
    test_results.write('test acc: ' + str(test_acc) + '\n')
    test_results.close()

    torch.save(best_model.state_dict(), save_path + '/output/cnn_model.pth')
    perf_data = np.array([train_losses, train_accs, val_losses, val_accs])

    return perf_data
