import torch
import torch.nn as nn
from typing import List, Dict

from vocoder.base.base_model import BaseModel
from vocoder.GEN import Generator
from vocoder.discriminator.mpd import MPD
from vocoder.discriminator.msd import MSD


class HiFiGAN(BaseModel):
    def __init__(self,
                 generator_config: Dict,
                 mpd_config: Dict,
                 msd_config: Dict):
        super().__init__()
        self.generator = Generator(**generator_config)
        self.mpd = MPD(**mpd_config)
        self.msd = MSD(**msd_config)