import torch
import torch.nn as nn
import numpy as np

from config import args
from models.my_net import MyNet
from data.my_dataset import MyDataset
from utils.losses import MyLoss
from utils.general import evaluate



np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")
kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}

data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
testData = MyDataset(dataset="test", transforms=data_transforms)
testLoader = DataLoader(testData, batch_size=args["BATCH_SIZE"], shuffle=True, **kwargs)

model = MyNet().to(device)
loss_function = MyLoss()



if args["TRAINED_MODEL_FILE"] is not None:

    print("\n\nTrained Model File: %s" %(args["TRAINED_MODEL_FILE"]))
    print("\nTesting the trained model .... \n")

    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"]))
    model.to(device)
    testLoss, testMetric = evaluate(model, testLoader, loss_function, device)
    
    print("Test Loss: %.6f, Test Metric: %.3f" %(testLoss, testMetric))
    print("\nTesting Done.\n")    
