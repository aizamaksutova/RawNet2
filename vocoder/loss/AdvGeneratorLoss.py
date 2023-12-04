import torch
from torch import Tensor
import torch.nn as nn

class GeneratorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_outputs):
        total_loss = 0.0
        for pred_output in pred_outputs:
            pred_loss = torch.mean((pred_output - 1) ** 2)
            total_loss += pred_loss
        return total_loss