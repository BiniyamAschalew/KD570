import typing as tp
import yaml
import fire
import torch

# classes camel casing, functions snake casing ( lower case), constants all caps snake casing
# lets use typing on everyfunction

from models import build_model
from train import train_model, test_model, train_distillation_model
from dataset import load_dataset
from utils import get_device, get_logger, load_config


def main(config_dir: str):

    config = load_config(config_dir)
    device = get_device(config["device"])
    logger = get_logger(log_dir=config["log_dir"])

    trainloader, testloader = load_dataset(config["dataset"], config["batch_size"])

    teacher_model = build_model(config["teacher_model"], device)

    student_model = build_model(config["student_model"], device)


    









if __name__ == "__main__":
    fire.Fire(main)