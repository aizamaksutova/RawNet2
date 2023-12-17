import numpy as np
from RawNet2.metrics.util_metrics import compute_eer


class EER:
    def __init__(self):
        self.name = "equal_error_rate"

    def __call__(self, target: np.array, pred: np.array, **kwargs):
        return compute_eer(pred[target == 1], pred[target == 0])[0]