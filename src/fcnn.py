import copy
import time
import torch
import torch.nn as nn
from nn_utils import train
from nn_utils import validate

class FCNN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(FCNN, self).__init__()

        self.fc1 = nn.Linear(observation_space, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_space)

    def forward(self, x):
        outs = nn.functional.relu(self.fc1(x))
        outs = nn.functional.relu(self.fc2(outs))
        outs = self.fc3(outs)
        return outs

def fcnn_train(model, batch_size, epochs, learning_rate, weight_decay):

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
