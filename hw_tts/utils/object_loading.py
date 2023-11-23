from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import hw_tts.datasets
from hw_tts.collate_fn.collate import collate_fn_tensor
from hw_tts.utils.parse_config import ConfigParser


class MyCollator():
    def __init__(self, batch_expand_size):
        self.batch_size = batch_expand_size

    def __call__(self, batch):
        return collate_fn_tensor(batch, self.batch_size)


def get_dataloader(configs: ConfigParser):
    params = configs["data"]
    dataset = configs.init_obj(params["dataset"], hw_tts.datasets)
    collator = MyCollator(params.get("batch_expand_size"))
    dataloader = DataLoader(
        dataset,
        batch_size=params.get("batch_expand_size") * params.get("batch_size"),
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=params.get("num_workers")
    )

    return dataloader