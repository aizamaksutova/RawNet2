import torch.nn.functional as F
from torch import distributions
from torch import nn
import torch
import numpy as np


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        x = x.transpose(self.dim_1, self.dim_2)
        return x


class VariancePredictor(nn.Module):
    def __init__(self,
        enc_dim, duration_predictor_filter_sz, duration_predictor_kernel_sz, dropout=0.1):
        super(VariancePredictor, self).__init__()

        self.input_size = enc_dim
        self.filter_size = duration_predictor_filter_sz
        self.kernel = duration_predictor_kernel_sz
        self.conv_output_size = duration_predictor_filter_sz
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out