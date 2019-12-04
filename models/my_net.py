import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):

    """
    space for documentation
    """

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)
        self.drop1 = nn.Dropout2d()
        self.bn1 = nn.BatchNorm2d()
        self.fc1 = nn.Linear(16*2*2, 6)
        self.weights_init()
        return

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return

    def forward(self, inputBatch):
        batch = F.relu(self.bn1(self.conv1(inputBatch)))
        batch = self.maxpool1(batch)
        batch = self.fc1(batch)
        outputBatch = self.drop1(batch)
        return outputBatch

