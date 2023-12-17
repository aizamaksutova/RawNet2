import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from RawNet2.resblock.fms import FMS



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, is_first=False):
        super().__init__()

        layers_before_skip = []
        if not is_first:
            layers_before_skip += [nn.BatchNorm1d(in_channels), nn.LeakyReLU(0.3)]
        layers_before_skip += [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same"),
        ]
        self.layers_before_skip = nn.Sequential(*layers_before_skip)
        self.channels_changer = None if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.layers_after_skip = nn.Sequential(
            nn.MaxPool1d(3),
            FMS(out_channels),
        )

    def forward(self, x):
        out = self.layers_before_skip(x)
        if self.channels_changer:
            x = self.channels_changer(x)
        out = out + x
        out = self.layers_after_skip(out)
        return out