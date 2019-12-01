import torch
import torch.nn as nn


class MyLoss(nn.Module):

    """
    Space for documentation
    """

    def __init__(self, lambd):
        super(MyLoss, self).__init__()
        self.lambd = lambd
        self.main_loss = nn.MSELoss()
        return


    def l2_regularization(self, model):
        loss = 0
        for name, p in model.named_parameters():
            if "bias" not in name:
                loss = loss + torch.sum(p*p)
        return loss

       
    def forward(self, out, target, model):
        regLoss = self.l2_regularization(model)
        mainLoss = self.main_loss(out, target)
        loss = mainLoss + self.lambd*regLoss
        return loss



if __name__ == '__main__':

    #code for testing the loss function