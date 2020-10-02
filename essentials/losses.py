import torch
import torch.nn as nn



class MyLoss(nn.Module):

    """
    Documentation: description, input, output
    """

    def __init__(self):
        super(MyLoss, self).__init__()
        return


    def forward(self, outputBatch, targetBatch):
        return loss



class L2Regularizer(nn.Module):

    """
    Documentation: description, input, output
    """

    def __init__(self, lambd):
        super(L2Regularizer, self).__init__()
        self.lambd = lambd
        return

    def forward(self, model):
        loss = 0
        for name, p in model.named_parameters():
            if "bias" not in name:
                loss = loss + torch.sum(p*p)
        loss = self.lambd*loss
        return loss
