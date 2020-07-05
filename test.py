import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

from config import args
from data.my_dataset import MyDataset
from models.my_net import MyNet
from essentials.losses import MyLoss, L2Regularizer
from essentials.core import evaluate



random.seed(args["SEED"])
np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")
kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}



if args["TRAINED_WEIGHTS_FILE"] is not None:

    testData = MyDataset("test", datadir=args["DATA_DIRECTORY"])
    testLoader = DataLoader(testData, batch_size=args["BATCH_SIZE"], shuffle=True, **kwargs)


    print("Trained Weights File: %s" %(args["TRAINED_WEIGHTS_FILE"]))

    model = MyNet()
    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_WEIGHTS_FILE"], map_location=device))
    model.to(device)

    loss_function = MyLoss()
    regularizer = L2Regularizer(lambd=args["LAMBDA"])


    print("Testing the trained model ....")

    testLoss, testMetric = evaluate(model, testLoader, loss_function, regularizer, device, testParams)

    print("Test Loss: %.6f || Test Metric: %.3f" %(testLoss, testMetric))
    print("Testing Done.")


else:
    print("Path to the trained weights file not specified.")
