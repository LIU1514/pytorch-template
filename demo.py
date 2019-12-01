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

            sample, label = prepare_input(file)
            sample = data_transforms(sample)
            inp = sample.view(1, sample.size(0), sample.size(1))
            target = label.view(1, label.size(0))

            inp, target = (inp.float()).to(device), target.to(device)    
            with torch.no_grad():
                out = model(inp)
            prediction = decode(out)

        print("File: %s" %(os.path.join(root + file)))
        print("Prediction: %s" %(prediction))
        print("Target: %s" %(target))
        print("\n")


print("\nDemo Completed.\n")
