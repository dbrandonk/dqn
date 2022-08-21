from torch import nn

LAYER_WIDTH = 64

class LunarLanderNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(LunarLanderNet, self).__init__()

        self.fc1 = nn.Linear(observation_space.shape[0], LAYER_WIDTH)
        self.fc2 = nn.Linear(LAYER_WIDTH, LAYER_WIDTH)
        self.fc3 = nn.Linear(LAYER_WIDTH, action_space)

    def forward(self, data):
        outs = nn.functional.relu(self.fc1(data))
        outs = nn.functional.relu(self.fc2(outs))
        outs = self.fc3(outs)
        return outs
