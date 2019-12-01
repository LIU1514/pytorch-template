import torch
import numpy as np


def show_sample(data):
    
    """
    function to print a random sample from the dataset
    """
    
    numSamples = len(data)
    ix = int((numSamples-1)*np.random.rand())
    sample, label = data[ix]
    print(sample)
    print(label)
    print(sample.size(), label.size())
    return



def prepare_input(file):
    
    """
    space for documentation
    """
    
    return sample, label


if __name__ == '__main__':
    
    #testing code