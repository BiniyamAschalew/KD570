import typing as tp
import yaml

import fire
import torch

# classes camel casing, functions snake casing ( lower case), constants all caps snake casing
# lets use typing on everyfunction

from models import ConvNetBuilder, ResNetBuilder
from train import train_model, test_model, train_distillation_model
from dataset import MNISTDataLoader, NoiseDataLoader
from utils import get_device, get_logger


def main(config_dir: str):

    with open(config_dir , "r", encoding = "utf-8") as file:
        config = yaml.safe_load(file)

    device = get_device()
    logger = get_logger(log_dir=config["log_dir"])





if __name__ == "__main__":
    fire.Fire(main)