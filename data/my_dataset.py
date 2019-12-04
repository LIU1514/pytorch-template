import torch
from torch.utils.data import Dataset
import numpy as np

from utils import prepare_input



class MyDataset(Dataset):

    """
    Space for documentation
    """
    
    def __init__(self, datapath, dataset, transforms=None):
        super(MyDataset, self).__init__()
        with open(dataDir + "/" + dataset + ".txt", "r") as f:
            lines = f.readlines()
        self.datalist = [dataDir + "/main/" + line.strip().split(" ")[0] for line in lines]
        self.transforms = transforms
        return
        

    def __getitem__(self, index):
        file = self.datalist[index]
        inp, trgt = prepare_input(file)
        if self.transform is not None:
            inp = self.transform(inp)
        return inp, trgt


    def __len__(self):
        return len(self.datalist)


