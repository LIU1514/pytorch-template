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

    for batch, (inputBatch, targetBatch) in enumerate(trainLoader):
        
        inputBatch, targetBatch = (inputBatch.float()).to(device), targetBatch.to(device)
        
        optimizer.zero_grad()
        outputBatch = model(inputBatch)
        loss = loss_function(outputBatch, targetBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        predictionBatch = decode(outputBatch.detach())
        trainingMetric = trainingMetric + compute_metric(predictionBatch, targetBatch)
    
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
        for batch, (inputBatch, targetBatch) in enumerate(evalLoader):
            
            inputBatch, targetBatch = (inputBatch.float()).to(device), targetBatch.to(device)
            
            outputBatch = model(inputBatch)
            loss = loss_function(outputBatch, targetBatch)

            evalLoss = evalLoss + loss.item()
            predictionBatch = decode(outputBatch)
            evalMetric = evalMetric + compute_metric(predictionBatch, targetBatch)

    evalLoss = evalLoss/len(evalLoader)
    evalMetric = evalMetric/len(evalLoader)
    return evalLoss, evalMetric

