import torch
import torch.nn as nn
import numpy as np

from config import args
from models.my_net import MyNet
from data.utils import prepare_input
from utils.decoders import decode


np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")



print("\nRunning Demo .... \n")
print("Trained Model File: %s\n" %(args["TRAINED_MODEL_FILE"]))
print("Demo Directory: %s\n\n" %(args["CODE_DIRECTORY"] + "/demo"))


model = MyNet().to(device)
model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"]))
model.to(device)
model.eval()
data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])


for root, dirs, files in os.walk(args["CODE_DIRECTORY"] + "/demo"):
    for file in files:
        if file.endswith(".ext"):

            inp, trgt = prepare_input(file)
            inp = data_transforms(inp)
            inputBatch = inp.view(1, inp.size(0), inp.size(1))
            targetBatch = trgt.view(1, trgt.size(0))

            inputBatch, targetBatch = (inputBatch.float()).to(device), targetBatch.to(device)    
            with torch.no_grad():
                outputBatch = model(inputBatch)
            predictionBatch = decode(outputBatch)
            pred = predictionBatch[0]

        print("File: %s" %(os.path.join(root + file)))
        print("Prediction: %s" %(pred))
        print("Target: %s" %(trgt))
        print("\n")


print("\nDemo Completed.\n")
