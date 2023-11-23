from hw_tts.encoder.encoder import Encoder
from hw_tts.variance_adaptor.adaptor import VarianceAdaptor
from hw_tts.waveform_decoder.decoder import Decoder

import torch
from torch import nn

from hw_tts.masks.masking import get_mask_from_lengths


class FastSpeech2(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = Encoder(**kwargs)

        self.variance_adaptor = VarianceAdaptor(**kwargs)
        self.decoder = Decoder(**kwargs)

        self.mel_linear = nn.Linear(kwargs["encoder_dim"], kwargs["num_mels"])

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0, pitch_target=None, p_alpha=1.0, energy_target=None, e_alpha=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        
        if self.training:
            output, duration_predictor_output, pitch_prediction, energy_prediction = self.variance_adaptor(x, alpha, length_target, p_alpha, pitch_target, e_alpha, energy_target, mel_max_length)
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_predictor_output, pitch_prediction, energy_prediction
        else:
            output, mel_pos = self.variance_adaptor(x, alpha, p_alpha=p_alpha, e_alpha=e_alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output