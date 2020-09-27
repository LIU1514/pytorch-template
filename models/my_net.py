"""
Author: Smeet Shah

Description: DL model class definition.

- MyNet model architecture with forward pass function
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):

    """
    Documentation: Task, Architecture, Input, Output
    """

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(8 * 2 * 2, 8)
        self.weights_init()
        return

    def weights_init(self):
        """
        Function to initialize network weights
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(
                    m.weight, gain=nn.init.calculate_gain("relu")
                )
        return

    def forward(self, inputBatch):
        """
        Forward pass function
        """
        batch = F.relu(self.bn1(self.conv1(inputBatch)))
        batch = self.maxpool1(batch)
        batch = self.drop1(batch)
        outputBatch = self.fc1(batch)
        return outputBatch


if __name__ == "__main__":
    sys.exit()
