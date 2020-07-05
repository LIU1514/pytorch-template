import torch
import random
import numpy as np
import os

from config import args
from models.my_net import MyNet
from data.tools import prepare_input
from essentials.decoders import decode
from essentials.metrics import compute_metric
from essentials.pprocs import preprocess_sample



random.seed(args["SEED"])
np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")



if args["TRAINED_WEIGHTS_FILE"] is not None:

    print("Trained Weights File: %s" %(args["TRAINED_WEIGHTS_FILE"]))
    print("Demo Directory: %s" %(args["DEMO_DIRECTORY"]))

    model = MyNet()
    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_WEIGHTS_FILE"], map_location=device))
    model.to(device)


    print("Running Demo ....")

    for root, dirs, files in os.walk(args["DEMO_DIRECTORY"]):
        for file in files:

            preprocess_sample(file)

            inp, trgt = prepare_input(file)
            inputBatch = torch.unsqueeze(inp, dim=0)
            targetBatch = torch.unsqueeze(trgt, dim=0)

            inputBatch, targetBatch = (inputBatch.float()).to(device), targetBatch.to(device)

            model.eval()
            with torch.no_grad():
                outputBatch = model(inputBatch)

            predictionBatch = decode(outputBatch)
            metricValue = compute_metric(predictionBatch, targetBatch)

            pred = predictionBatch[0][:]
            trgt = targetBatch[0][:]

            print("File: %s" %(file))
            print("Prediction: %s" %(pred))
            print("Target: %s" %(trgt))
            print("Metric: %.3f" %(metricValue))
            print("\n")

    print("Demo Completed.")


else:
    print("Path to trained weights file not specified.")
