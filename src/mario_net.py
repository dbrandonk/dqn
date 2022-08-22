from torch import nn
import torch

class MarioNet(nn.Module):

    def __init__(self, observation_space, action_space):
        super(MarioNet, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.make_flat = nn.Flatten()
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, input):
        outs = input.float()
        outs = outs / 255.0
        outs = nn.functional.relu(self.cnn1(outs))
        outs = nn.functional.relu(self.cnn2(outs))
        outs = nn.functional.relu(self.cnn3(outs))
        outs = self.make_flat(outs)
        outs = nn.functional.relu(self.fc1(outs))
        outs = self.fc2(outs)

        return outs
