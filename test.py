import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import RawNet2.model as module_model
from RawNet2.utils.parse_config import ConfigParser


def main(config, args):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    logger.info("Checkpoint has been loaded.")
    model = model.to("cpu")
    model.eval()

    cut_length = config["data"]["train"]["datasets"][0]["args"]["max_sec_length"] * 16000

    for audio_name in sorted(os.listdir(args.inference_directory)):
        audio, sr = torchaudio.load(f"{args.inference_directory}/{audio_name}")
        resampled_audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
        pred = model(resampled_audio[:cut_length])["pred"][0].detach().cpu().numpy()
        verdict = "bonafide (real person)" if abs(pred[0]) > abs(pred[1]) else "spoofed"
        logger.info(f"{audio_name}:\nverdict: {verdict}\t\tpredictions: {pred}\n")

    logger.info("Testing has ended.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-c",
        "--config",
        default="config.json",
        type=str,
        help="The config used while training.",
    )
    args.add_argument(
        "-m",
        "--model_checkpoint",
        default="checkpoint.pth",
        type=str,
        help="Checkpoint path.",
    )
    args.add_argument(
        "-inf",
        "--inference_directory",
        default="inference/",
        type=str,
        help="Audio for inference data",
    )
    args = args.parse_args()

    model_config = Path(args.config)
    with model_config.open() as fin:
        config = ConfigParser(json.load(fin))

    main(config, args)