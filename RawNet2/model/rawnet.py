import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from RawNet2.base.base_model import BaseModel
from RawNet2.sincfilter import SincFilter
from RawNet2.resblock import ResBlock


class RawNet2(BaseModel):
    def __init__(
        self,
        sinc_channels,
        sinc_filter_length,
        channels1,
        channels2,
        gru_hidden_size,
        sinc_filter_type,
        requires_grad=True,
    ):
        super().__init__()

        self.sinc_filters = SincFilter(
            sinc_channels,
            sinc_filter_length,
            sinc_filter_type,
            requires_grad,
        )
        self.pre_resblocks = nn.Sequential(
            nn.MaxPool1d(3),
            nn.BatchNorm1d(sinc_channels),
            nn.LeakyReLU(0.3),
        )
        self.resblocks = nn.Sequential(
            ResBlock(sinc_channels, channels1, 3, True),
            ResBlock(channels1, channels2, 3),
            *[ResBlock(channels2, channels2, 3) for _ in range(4)],
            nn.BatchNorm1d(channels2),
            nn.LeakyReLU(0.3),
        )
        self.gru = nn.GRU(channels2, gru_hidden_size, num_layers=3, batch_first=True)
        self.head = nn.Linear(gru_hidden_size, 2)

    def forward(self, audio, **kwargs):
        x = audio.unsqueeze(1)
        x = self.sinc_filters(x)
        x = self.pre_resblocks(x)
        x = self.resblocks(x)
        x = self.gru(x.transpose(1, 2))[0][:, -1, :]
        return {"pred": self.head(x)}