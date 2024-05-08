import typing as tp
import yaml
import fire
import torch
import torch.nn as nn
# import dataparallel


# classes camel casing, functions snake casing ( lower case), constants all caps snake casing
# lets use typing on everyfunction

from models import build_model
from train import train_model, test_model, train_distillation_model
from dataset import load_dataset
from utils import get_device, get_logger, load_config, set_seed


def main(config_dir: str):

    config = load_config(config_dir)
    device = get_device(config["device"])
    logger = get_logger(log_dir=config["log_dir"])
    parallel_process = config["multiprocess"]

    set_seed(config["seed"])

    trainloader, testloader = load_dataset(config["dataset"], config["batch_size"])

    teacher_config = load_config(config["teacher_model_config"])
    student_config = load_config(config["student_model_config"])

    teacher_model = build_model(teacher_config, device, parallel_process)
    teacher_model = nn.DataParallel(teacher_model)#.to(device)
    logger.print(f"Teacher model successfully loaded")

    student_model1 = build_model(student_config, device, parallel_process)
    student_model2 = build_model(student_config, device, parallel_process)
    
    ## to make the students weight identical
    student_model2.load_state_dict(student_model1.state_dict())

    student_model1 = nn.DataParallel(student_model1)
    student_model2 = nn.DataParallel(student_model2)#.to(device) <- applied inside build_model function

    assert torch.norm(student_model1.module.conv1.weight - student_model2.module.conv1.weight) == 0
    logger.print(f"Student model successfully loaded")

    """Training the models"""

    criterion = nn.CrossEntropyLoss()



# def train_model(model: nn.Module, trainloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int, device: torch.device, logger: object):

    if not teacher_config["load_from_path"]:
        teacher_model = train_model(teacher_model, trainloader, config["teacher_model_config"], device, logger)
        torch.save(teacher_model.state_dict(), config["teacher_model_path"])

    



















if __name__ == "__main__":
    fire.Fire(main)