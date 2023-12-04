import torch
from torch import nn
import torch.nn.functional as F

from vocoder.loss.AdvDiscriminatorLoss import DiscriminatorAdvLoss
from vocoder.loss.AdvGeneratorLoss import GeneratorAdvLoss
from vocoder.loss.MelSpecLoss import MelSpectrogramLoss
from vocoder.loss.FeatMatchingLoss import FeatureMatchingLoss


class HiFiGANLoss(nn.Module):
    def __init__(self,
                 feature_matching_multiplier,
                 mel_spectrogram_multiplier):
        super().__init__()
        self.discriminator_adv_loss = DiscriminatorAdvLoss()
        self.generator_adv_loss = GeneratorAdvLoss()
        self.mel_spectrogram_loss = MelSpectrogramLoss(mel_spectrogram_multiplier)
        self.feature_matching_loss = FeatureMatchingLoss(feature_matching_multiplier)