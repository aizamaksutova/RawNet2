import torch
import torch.nn as nn
from typing import List

from vocoder.base.base_model import BaseModel
from vocoder.GEN.mrf import MRF


class GeneratorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_trans_kernel_size: int,
                 mrf_kernel_sizes: List[int],
                 mrf_dilations: List[List[List[int]]]):

        super().__init__()
        
        self.sequential = nn.Sequential(

            nn.LeakyReLU(),
            nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=conv_trans_kernel_size,
                    stride=conv_trans_kernel_size // 2,
                    padding=(conv_trans_kernel_size - conv_trans_kernel_size // 2) // 2
                )
            ),
            MRF(
                num_channels=in_channels // 2,
                kernel_sizes=mrf_kernel_sizes,
                dilations=mrf_dilations
            )
        )

    def forward(self, x):
        return self.sequential(x)


class Generator(BaseModel):
    def __init__(self,
                 in_channels: int,
                 hidden_dimension: int,
                 conv_kernel_sz: List[int],
                 mrf_kernel_sz: List[int],
                 mrf_dilations: List[List[List[int]]]):

        super().__init__()
        
        self.start = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dimension,
                kernel_size=7,
                dilation=1,
                padding="same"
            )
        )
        self.stack = nn.ModuleList([
            GeneratorBlock(
                in_channels=hidden_dimension // (2 ** i),
                conv_trans_kernel_size=conv_kernel_sz[i],
                mrf_kernel_sizes=mrf_kernel_sz,
                mrf_dilations=mrf_dilations
            )
            for i in range(len(conv_kernel_sz))
        ])
        end_channels = hidden_dimension // (2 ** len(conv_kernel_sz))
        self.end = nn.Sequential(
            nn.LeakyReLU(),
            nn.utils.weight_norm(
                nn.Conv1d(
                    in_channels=end_channels,
                    out_channels=1,
                    kernel_size=7,
                    padding="same"
                )
            ),
            nn.Tanh()
        )

    def forward(self, x):
        res = self.start(x)
        for block in self.stack:
            res = block(res)
        res = self.end(res)
        return res
    
    def remove_normalization(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose1d):
                nn.utils.remove_weight_norm(module)