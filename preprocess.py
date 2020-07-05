import torch
import numpy as np
from tqdm import tqdm
import random
import os

from config import args
from essentials.pprocs import preprocess_sample



random.seed(args["SEED"])
np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")


filesList = list()
for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
    for file in files:
        filesList.append(os.path.join(root, file))


print("Number of data samples to be processed = %d" %(len(filesList)))
print("Starting preprocessing ....")

for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
    preprocess_sample(file)

print("Preprocessing Done.")
