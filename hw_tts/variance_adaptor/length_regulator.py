import torch
import torch.nn as nn
import torch.nn.functional as F


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

def length_regulator(x, duration_predictor_output, mel_max_length=None):
    expand_max_len = torch.max(
        torch.sum(duration_predictor_output, -1), -1)[0]
    alignment = torch.zeros(duration_predictor_output.size(0),
                            expand_max_len,
                            duration_predictor_output.size(1)).numpy()
    alignment = create_alignment(alignment,
                                    duration_predictor_output.cpu().numpy())
    alignment = torch.from_numpy(alignment).to(x.device)

    output = alignment @ x
    if mel_max_length:
        output = F.pad(
            output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
    return output