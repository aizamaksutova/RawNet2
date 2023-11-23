import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_tts.masks.masking import get_non_pad_mask, get_attn_key_pad_mask

from hw_tts.encoder.encoder import FFTBlock



class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, **kwargs):

        super().__init__()

        n_position = kwargs["max_seq_len"] + 1
        n_layers = kwargs["decoder_n_layer"]

        self.position_enc = nn.Embedding(
            n_position,
            kwargs["encoder_dim"],
            padding_idx=kwargs["PAD"],
        )

        self.layer_stack =nn.ModuleList([FFTBlock(
            kwargs["encoder_dim"],
            kwargs["encoder_conv1d_filter_sz"],
            kwargs["encoder_head"],
            kwargs["encoder_dim"] // kwargs["encoder_head"],
            kwargs["encoder_dim"] // kwargs["encoder_head"],
            kwargs["fft_conv1d_kernel"],
            kwargs["fft_conv1d_padding"],
            dropout=kwargs["dropout"]
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output