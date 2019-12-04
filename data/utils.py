import torch
import numpy as np


def show_sample(data):
    
    """
    function to print a random sample from the dataset
    """
    
    numSamples = len(data)
    ix = int((numSamples-1)*np.random.rand())
    inp, trgt = data[ix]
    print(inp)
    print(trgt)
    print(inp.size(), trgt.size())
    return



def prepare_input(file):
    
    """
    space for documentation
    """
    
    return inp, trgt

