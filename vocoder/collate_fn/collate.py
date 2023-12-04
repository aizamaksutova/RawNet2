import numpy as np
import torch
import torch.nn.functional as F

from vocoder.utils import pad_1D_tensor, pad_2D_tensor



def collate(batch):
    output = {}
    output["wav_gt"] = pad_1D_tensor([item["wav_gt"].squeeze(0) for item in batch]).unsqueeze(1)
    output["mel_gt"] = pad_2D_tensor([item["mel_gt"] for item in batch])
    return output