import torch.nn.functional as F
from torch import distributions
from torch import nn
import torch
import numpy as np

from .predictor import VariancePredictor
from .length_regulator import length_regulator



class VarianceAdaptor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.energy = VariancePredictor(kwargs["encoder_dim"], kwargs["duration_predictor_filter_sz"], 
        kwargs["duration_predictor_kernel_sz"], kwargs["dropout"])

        self.pitch = VariancePredictor(kwargs["encoder_dim"], kwargs["duration_predictor_filter_sz"], 
        kwargs["duration_predictor_kernel_sz"], kwargs["dropout"])

        self.duration = VariancePredictor(kwargs["encoder_dim"], kwargs["duration_predictor_filter_sz"], 
        kwargs["duration_predictor_kernel_sz"], kwargs["dropout"])

        self.num_bins = kwargs["num_bins"]

        energy_min, energy_max = kwargs["energy_stat"][0], kwargs["energy_stat"][1]

        pitch_min, pitch_max = kwargs["pitch_stat"][0], kwargs["pitch_stat"][1]


        self.pitch_bins = nn.Parameter(
            torch.linspace(pitch_min, pitch_max, self.num_bins - 1),
            requires_grad=False,
        )

        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, self.num_bins - 1),
            requires_grad=False,
        )
        
        self.pitch_embedding = nn.Embedding(
            self.num_bins, kwargs["encoder_dim"]
        )
        self.energy_embedding = nn.Embedding(
            self.num_bins, kwargs["encoder_dim"]
        )

    
    def forward(self, x, alpha=1.0, target=None, p_alpha=1.0, p_target=None, e_alpha=1.0, e_target=None, mel_max_length=None):
        duration_predictor_output = self.duration(x)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if target is not None:
            output  = length_regulator(x, target, mel_max_length)
            pitch_prediction = self.pitch(output)
            energy_prediction = self.energy(output)
            
            p_embedding = self.pitch_embedding(torch.bucketize(p_target, self.pitch_bins))
            e_embedding = self.energy_embedding(torch.bucketize(e_target, self.energy_bins))
            
            output = output + p_embedding + e_embedding
            return output, duration_predictor_output, pitch_prediction, energy_prediction
        else:
            duration_predictor_output = torch.expm1(duration_predictor_output)
            duration_predictor_output = (duration_predictor_output * alpha + 0.5).int()
            
            output = length_regulator(x, duration_predictor_output)
            pitch_prediction = torch.expm1(self.pitch(output)) * p_alpha
            energy_prediction = torch.expm1(self.energy(output)) * e_alpha
            
            p_embedding = self.pitch_embedding(torch.bucketize(pitch_prediction, self.pitch_bins))
            e_embedding = self.energy_embedding(torch.bucketize(energy_prediction, self.energy_bins))
            output = output + p_embedding + e_embedding
            
            mel_pos = torch.stack([torch.Tensor([i+1 for i in range(output.size(1 ))])]).long().to(device)
            return output, mel_pos


