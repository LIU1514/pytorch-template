import torch
import numpy as np

from metrics import compute_metric
from decoders import decode


def num_params(model):
    
    """
    space for documentation
    """

    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams



def train(model, trainLoader, optimizer, loss_function, device):
    
    """
    space for documentation
    """

    model.train()
    trainingLoss = 0
    trainingMetric = 0

    for batch, (inp, target) in enumerate(trainLoader):
        
        inp, target = (inp.float()).to(device), target.to(device)
        
        optimizer.zero_grad()
        out = model(inp)
        loss = loss_function(out, target)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        prediction = decode(out.detach())
        trainingMetric = trainingMetric + compute_metric(prediction, target)
    
    trainingLoss = trainingLoss/len(trainLoader)
    trainingMetric = trainingMetric/len(trainLoader)
    return trainingLoss, trainingMetric



def evaluate(model, evalLoader, loss_function, device):
    
    """
    space for documentation
    """
    
    model.eval()
    evalLoss = 0
    evalMetric = 0
    
    with torch.no_grad():
        for batch, (inp, target) in enumerate(evalLoader):
            
            inp, target = (inp.float()).to(device), target.to(device)
            
            out = model(inp)
            loss = loss_function(out, target)

            evalLoss = evalLoss + loss.item()
            prediction = decode(out)
            evalMetric = evalMetric + compute_metric(prediction, target)

    evalLoss = evalLoss/len(evalLoader)
    evalMetric = evalMetric/len(evalLoader)
    return evalLoss, evalMetric




if __name__ == '__main__':
    
    #code for testing the functions