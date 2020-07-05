from torch.utils.data import Dataset
import numpy as np

from .tools import prepare_input



class MyDataset(Dataset):

    """
    Documentation: description, input, output
    """

    def __init__(self, dataset, datadir):
        super(MyDataset, self).__init__()
        self.datalist = datalist
        return


    def __getitem__(self, index):
        file = self.datalist[index]
        inp, trgt = prepare_input(file)
        return inp, trgt


    def __len__(self):
        return len(self.datalist)
