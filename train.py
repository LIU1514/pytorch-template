import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

from config import args
from models.my_net import MyNet
from data.my_dataset import MyDataset
from utils.losses import MyLoss
from utils.general import num_params, train, evaluate



matplotlib.use("Agg")
np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")
kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}


data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
trainData = MyDataset(dataset="train", transforms=data_transforms)
valDataSize = int(args["VALIDATION_SPLIT"]*len(trainData))
trainDataSize = len(trainData) - valDataSize
trainData, valData = random_split(trainData, [trainDataSize, valDataSize])

trainLoader = DataLoader(trainData, batch_size=args["BATCH_SIZE"], shuffle=True, **kwargs)
valLoader = DataLoader(valData, batch_size=args["BATCH_SIZE"], shuffle=True, **kwargs)



model = MyNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=args["LEARNING_RATE"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args["LR_DECAY"])
loss_function = MyLoss()



if os.path.exists(args["CODE_DIRECTORY"] + "/checkpoints"):
    
    while True:
        ch = input("Continue and remove the 'checkpoints' directory? y/n: ")
        if ch == "y":
            break
        elif ch == "n":
            exit()
        else:
            print("Invalid input")
    shutil.rmtree(args["CODE_DIRECTORY"] + "/checkpoints")

os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints")
os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/models")
os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/plots")



if args["PRETRAINED_MODEL_FILE"] is not None:

    print("\n\nPre-trained Model File: %s" %(args["PRETRAINED_MODEL_FILE"]))
    print("\nLoading the pre-trained model .... \n")
    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["PRETRAINED_MODEL_FILE"]))
    model.to(device)
    print("\nLoading Done.\n")    



trainingLossCurve = list()
validationLossCurve = list()
trainingMetricCurve = list()
validationMetricCurve = list()


print("\nTraining the model .... \n")

numTotalParams, numTrainableParams = num_params(model)
print("Number of total parameters in the model = %d" %(numTotalParams))
print("Number of trainable parameters in the model = %d\n" %(numTrainableParams))

for epoch in range(1, args["NUM_EPOCHS"]+1):
    
    trainingLoss, trainingMetric = train(model, trainLoader, optimizer, loss_function, device)
    trainingLossCurve.append(trainingLoss)
    trainingMetricCurve.append(trainingMetric)

    validationLoss, validationMetric = evaluate(model, valLoader, loss_function, device)
    validationLossCurve.append(validationLoss)
    validationMetricCurve.append(validationMetric)

    scheduler.step()

    print("Epoch: %d \t Tr.Loss: %.6f \t Val.Loss: %.6f \t Tr.Metric: %.3f \t Val.Metric: %.3f" 
          %(epoch, trainingLoss, validationLoss, trainingMetric, validationMetric))
    

    if (epoch % args["SAVE_FREQUENCY"] == 0) or (epoch == args["NUM_EPOCHS"]):
        
        savePath = args["CODE_DIRECTORY"] + "/checkpoints/models/epoch_{:04d}-metric_{:.3f}.pt".format(epoch, validationMetric)
        torch.save(model.state_dict(), savePath)

        plt.figure()
        plt.title("Loss Curves")
        plt.xlabel("Epoch No.")
        plt.ylabel("Loss value")
        plt.plot(list(range(1, len(trainingLossCurve)+1)), trainingLossCurve, "blue", label="Train")
        plt.plot(list(range(1, len(validationLossCurve)+1)), validationLossCurve, "red", label="Validation")
        plt.legend()
        plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/epoch_{:04d}_loss.png".format(epoch))
        plt.close()

        plt.figure()
        plt.title("Metric Curves")
        plt.xlabel("Epoch No.")
        plt.ylabel("Metric")
        plt.plot(list(range(1, len(trainingMetricCurve)+1)), trainingMetricCurve, "blue", label="Train")
        plt.plot(list(range(1, len(validationMetricCurve)+1)), validationMetricCurve, "red", label="Validation")
        plt.legend()
        plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/epoch_{:04d}_metric.png".format(epoch))
        plt.close()


print("\nTraining Done.\n")
