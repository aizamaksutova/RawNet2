from operator import xor
from typing import List

import torch
import torch.nn.functional as F

from torch.utils.data import ConcatDataset, DataLoader

# import src.augmentations
import RawNet2.datasets

from RawNet2.utils.parse_config import ConfigParser

def pad_1D_tensor(inputs):
    def pad_data(x, length):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len) for x in inputs])

    return padded


def collate_fn(batch: List[dict]):
    audio = pad_1D_tensor([item["audio"].squeeze(0) for item in batch])
    target = torch.Tensor([item["target"] for item in batch]).to(torch.long)

    return {"audio": audio, "target": target}


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == "train":
            #     wave_augs, spec_augs = src.augmentations.from_configs(configs)
            drop_last = True
        else:
            #     wave_augs, spec_augs = None, None
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            # datasets.append(configs.init_obj(ds, src.datasets, config_parser=configs, wave_augs=wave_augs, spec_augs=spec_augs))
            datasets.append(configs.init_obj(ds, RawNet2.datasets, config_parser=configs))

        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            if shuffle in params.keys():
                shuffle = params["shuffle"]
            batch_sampler = None
        # elif "batch_sampler" in params:
        #     batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module, data_source=dataset)
        #     bs, shuffle = 1, False
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            drop_last=drop_last,
        )
        dataloaders[split] = dataloader
    return dataloaders