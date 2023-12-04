import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from vocoder.base.base_model import BaseModel

class subdisctiminator(nn.Module):
    def __init__(self,
                 period: int,
                 kernel_size: int,
                 stride: int,
                 channels: List[int]):
        super().__init__()
        self.period = period

        layers = []

        # Adding first input channel
        channels = [1] + channels

        for i in range(len(channels) - 1):
            layers.append(
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv2d(
                            in_channels=channels[i],
                            out_channels=channels[i + 1],
                            kernel_size=(kernel_size, 1),
                            stride=(stride, 1),
                            padding=((kernel_size - 1) // 2, 0)
                        )
                    ),
                    nn.LeakyReLU()
                )
            )
        
        layers.append(
            nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv2d(
                        in_channels=channels[-1],
                        out_channels=1024,
                        kernel_size=(5, 1),
                        padding="same"
                    )
                ),
                nn.LeakyReLU(),
                nn.utils.weight_norm(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1,
                    kernel_size=(3, 1),
                    padding="same"
                )
            )
            )
        )


        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        features_from_layers = []
        if x.shape[-1] % self.period > 0:
            x = F.pad(x, (0, self.period - x.shape[-1] % self.period), mode="reflect")
        x = x.reshape(x.shape[0], 1, x.shape[-1] // self.period, self.period)
        for layer in self.layers:
            x = layer(x)
            features_from_layers.append(x)
        return x.flatten(-2, -1), features_from_layers[:-1]


class MPD(BaseModel):
    def __init__(self,
                 periods: List[int],
                 kernel_size: int,
                 stride: int,
                 channels: List[int]):

        super().__init__()

        self.discriminators = nn.ModuleList([
            subdisctiminator(
                period=period,
                kernel_size=kernel_size,
                stride=stride,
                channels=channels
            )
            for period in periods
        ])
    def forward(self, x):
        mpd_outputs = []
        mpd_features = []
        for disc in self.discriminators:
            output, features_list = disc(x)
            mpd_outputs.append(output)
            mpd_features.append(features_list)
        return mpd_outputs, mpd_features

