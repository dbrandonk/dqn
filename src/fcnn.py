import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(FCNN, self).__init__()

        self.fc1 = nn.Linear(observation_space, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, action_space)

    def forward(self, x):
        outs = nn.functional.relu(self.fc1(x))
        outs = nn.functional.relu(self.fc2(outs))
        outs = self.fc3(outs)
        return outs
