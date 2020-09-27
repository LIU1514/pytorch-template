"""
Author: Smeet Shah

Description: Python script to train the DL model.

- Loads the training and validation datasets
- Initializes the model with pre-trained weights if available
- Trains the model over the training set for a fixed number of epochs
- Prints the number of total parameters and trainable parameters in the model
- Prints the loss and evalution metric values on both training and validation
sets after each epoch
- Intermediate weights and loss/metric plots are stored in a temporary
"checkpoints" directory
"""

import os
import sys
import shutil
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from config import args
from models import MyNet
from data import MyDataset
from essentials import num_params, train, evaluate
from essentials.losses import MyLoss, L2Regularizer


def main():

    """
    Main function wrapper for training script.
    """

    matplotlib.use("Agg")
    random.seed(args["SEED"])
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
        kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True}
    else:
        device = torch.device("cpu")
        kwargs = {}

    trainData = MyDataset("train", datadir=args["DATA_DIRECTORY"])
    valSize = int(args["VALIDATION_SPLIT"] * len(trainData))
    trainSize = len(trainData) - valSize
    trainData, valData = random_split(trainData, [trainSize, valSize])
    trainLoader = DataLoader(
        trainData, batch_size=args["BATCH_SIZE"], shuffle=True, **kwargs
    )
    valLoader = DataLoader(
        valData, batch_size=args["BATCH_SIZE"], shuffle=True, **kwargs
    )

    model = MyNet()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args["LEARNING_RATE"],
        betas=(args["MOMENTUM1"], args["MOMENTUM2"]),
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args["LR_DECAY"]
    )
    criterion = MyLoss()
    regularizer = L2Regularizer(lambd=args["LAMBDA"])

    if os.path.exists(args["CODE_DIRECTORY"] + "/checkpoints"):
        while True:
            char = input(
                "Continue and remove the 'checkpoints' directory? y/n: "
            )
            if char == "y":
                break
            if char == "n":
                sys.exit()
            else:
                print("Invalid input")
        shutil.rmtree(args["CODE_DIRECTORY"] + "/checkpoints")

    os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints")
    os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/plots")
    os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/weights")

    if args["PRETRAINED_WEIGHTS_FILE"] is not None:
        print(
            "Pretrained Weights File: %s" % (args["PRETRAINED_WEIGHTS_FILE"])
        )
        print("Loading the pretrained weights ....")
        model.load_state_dict(
            torch.load(
                args["CODE_DIRECTORY"] + args["PRETRAINED_WEIGHTS_FILE"],
                map_location=device,
            )
        )
        model.to(device)
        print("Loading Done.")

    trainingLossCurve = list()
    validationLossCurve = list()
    trainingMetricCurve = list()
    validationMetricCurve = list()

    numTotalParams, numTrainableParams = num_params(model)
    print("Number of total parameters in the model = %d" % (numTotalParams))
    print(
        "Number of trainable parameters in the model = %d"
        % (numTrainableParams)
    )

    print("Training the model ....")

    for epoch in range(1, args["NUM_EPOCHS"] + 1):

        trainingLoss, trainingMetric = train(
            model, trainLoader, optimizer, criterion, regularizer, device
        )
        trainingLossCurve.append(trainingLoss)
        trainingMetricCurve.append(trainingMetric)

        validationLoss, validationMetric = evaluate(
            model, valLoader, criterion, regularizer, device
        )
        validationLossCurve.append(validationLoss)
        validationMetricCurve.append(validationMetric)

        print(
            (
                "| Epoch: %03d |"
                "| Tr.Loss: %.6f  Val.Loss: %.6f |"
                "| Tr.Metric: %.3f  Val.Metric: %.3f |"
            )
            % (
                epoch,
                trainingLoss, validationLoss,
                trainingMetric, validationMetric,
            )
        )

        scheduler.step()

        if epoch % args["SAVE_FREQUENCY"] == 0:

            savePath = (
                args["CODE_DIRECTORY"]
                + "/checkpoints/weights/epoch_{:04d}-metric_{:.3f}.pt"
            ).format(epoch, validationMetric)
            torch.save(model.state_dict(), savePath)

            plt.figure()
            plt.title("Loss Curves")
            plt.xlabel("Epoch No.")
            plt.ylabel("Loss value")
            plt.plot(
                list(range(1, len(trainingLossCurve) + 1)),
                trainingLossCurve,
                "blue",
                label="Train",
            )
            plt.plot(
                list(range(1, len(validationLossCurve) + 1)),
                validationLossCurve,
                "red",
                label="Validation",
            )
            plt.legend()
            plt.savefig(
                (
                    args["CODE_DIRECTORY"]
                    + "/checkpoints/plots/epoch_{:04d}_loss.png"
                ).format(epoch)
            )
            plt.close()

            plt.figure()
            plt.title("Metric Curves")
            plt.xlabel("Epoch No.")
            plt.ylabel("Metric")
            plt.plot(
                list(range(1, len(trainingMetricCurve) + 1)),
                trainingMetricCurve,
                "blue",
                label="Train",
            )
            plt.plot(
                list(range(1, len(validationMetricCurve) + 1)),
                validationMetricCurve,
                "red",
                label="Validation",
            )
            plt.legend()
            plt.savefig(
                (
                    args["CODE_DIRECTORY"]
                    + "/checkpoints/plots/epoch_{:04d}_metric.png"
                ).format(epoch)
            )
            plt.close()

    print("Training Done.")

    return


if __name__ == "__main__":
    main()
