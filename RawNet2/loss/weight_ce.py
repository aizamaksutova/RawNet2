import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class CrossEntropyLoss_RawNet(nn.Module):
    def __init__(self, weight):
        super().__init__()
        weights = torch.tensor(weight, dtype=torch.float)
        self.ce_loss = CrossEntropyLoss(weight=weights)

    def forward(self, pred, target, **kwargs):

        return {"loss": self.ce_loss(pred, target)}