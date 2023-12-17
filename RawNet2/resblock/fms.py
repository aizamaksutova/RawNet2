import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from RawNet2.base.base_model import BaseModel
from RawNet2.sincfilter import SincFilter



class FMS(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.linear = nn.Linear(in_features=num_features, out_features=num_features)

    def forward(self, x):
        scale = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        scale = self.linear(scale)
        scale = F.sigmoid(scale).unsqueeze(-1)
        return x * scale + scale