import typing as tp
import yaml
import fire
import torch
import torch.nn as nn
# import dataparallel


# classes camel casing, functions snake casing ( lower case), constants all caps snake casing
# lets use typing on everyfunction

from models import load_model
from train import train_model, test_model, train_distillation_model, save_model
from dataset import load_dataset
from utils import get_device, get_logger, load_config, set_seed


def main(config_dir: str):

    config = load_config(config_dir)
    device = get_device(config["device"])
    logger = get_logger(log_dir=config["log_dir"])


    PARALLEL = config["multiprocess"]
    EPOCHS = config["epochs"]

    set_seed(config["seed"])

    train_loader, test_loader = load_dataset(config["dataset"], config["batch_size"])

    teacher_config = load_config(config["teacher_model_config"])
    student_config = load_config(config["student_model_config"])

    teacher_model = load_model(teacher_config, device, PARALLEL)
    teacher_model = nn.DataParallel(teacher_model)#.to(device)
    logger.print(f"Teacher model successfully loaded")

    student_model1 = load_model(student_config, device, PARALLEL)
    student_model2 = load_model(student_config, device, PARALLEL)
    
    ## to make the students weight identical
    student_model2.load_state_dict(student_model1.state_dict())

    # assert torch.norm(student_model1.module.conv_blocks.0.conv.weight - student_model2.module.conv_blocks.0.conv.weight) == 0
    logger.print(f"Student model successfully loaded")

    """Training the models"""

    logger.print("Training the models")

    logger.print(f"Training student 1 model: {student_config['model']}")    
    # student_model1 = train_model(student_model1, train_loader, EPOCHS, device, logger)
    # student_acc1 = test_model(student_model1, test_loader, device, logger)

    logger.print(f"Training student 2 model: {student_config['model']}")
    student_model2 = train_distillation_model(teacher_model, student_model2, train_loader, device, logger, config)
    student_acc2 = test_model(student_model2, test_loader, device, logger) 

    # if not teacher_config["load_from_path"]:
    #     teacher_model = train_model(teacher_model, trainloader, EPOCHS, device, logger)

    #     if teacher_config["save_path"]:
    #         torch.save(teacher_model.state_dict(), teacher_config["save_path"])
    #         logger.print(f"Teacher model saved at {teacher_config['save_path']}")



    

        
    



















if __name__ == "__main__":
    fire.Fire(main)