from torch import nn
import torchvision

class MarioNet(nn.Module):

    def __init__(self, observation_space, action_space):
        super().__init__()

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )

    def forward(self, input):
        input = input.unsqueeze(1)
        input = input.float()
        input = input / 255.0

        return self.online(input)
