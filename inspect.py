import torch
import numpy as np
import random

from config import args
from data.my_dataset import MyDataset
from models.my_net import MyNet
from essentials.losses import MyLoss, L2Regularizer
from essentials.decoders import decode
from essentials.metrics import compute_metric
from essentials.pprocs import preprocess_sample



def inspect_mynet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNet()
    model.to(device)
    N, C, H, W = args["BATCH_SIZE"], 3, 32, 32
    inputBatch = torch.rand(N, C, H, W).to(device)
    model.eval()
    with torch.no_grad():
        outputBatch = model(inputBatch)
    print(outputBatch.shape)
    return



def inspect_mydataset():
    trainData = MyDataset("train", datadir=args["DATA_DIRECTORY"])
    trainSize = len(trainData)
    ix = random.randint(0, trainSize-1)
    inp, trgt = trainData[ix]
    print(inp.shape, trgt.shape)
    return



def inspect_decode():
    return



def inspect_compute_metric():
    return



def inspect_preprocess_sample():
    return



def inspect_myloss():
    return



def inspect_l2regularizer():
    return



if __name__ == '__main__':
    #call the required inspect function
    exit()
