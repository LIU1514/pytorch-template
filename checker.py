import torch
import numpy as np

from config import args
from models.my_net import MyNet


def function1_name_checker():
    return


def function2_name_checker():
    return


def mynet_checker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNet().to(device)
    T, N, C = 42, 8, 321
    inputBatch = torch.rand(T, N, C).to(device)
    outputBatch = model(inputBatch)
    print(outputBatch.size())
    return


if __name__ == '__main__':
    #call the required checker function