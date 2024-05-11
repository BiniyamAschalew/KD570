import typing as tp
import yaml
import fire
import warnings

import pandas as pd
import torch
import torch.nn as nn
# import dataparallel

from models import load_model
from train import train_model, test_model, train_distillation_model, save_model
from dataset import load_dataset
from utils import get_device, get_logger, load_config, set_seed

warnings.filterwarnings("ignore")

def main(config_dir: str):

    config = load_config(config_dir)
    device = get_device(config["device"])
    logger = get_logger(log_dir=config["log_dir"])

    PARALLEL = config["multiprocess"]
    EPOCHS = config["epochs"]
    LEARNING_RATE = config["learning_rate"]

    set_seed(config["seed"])

    train_loader, test_loader = load_dataset(config["dataset"], config["batch_size"])

    teacher_config = load_config(config["teacher_model_config"])
    teacher_model, trained = load_model(teacher_config, device, PARALLEL)
    # teacher_model = nn.DataParallel(teacher_model).to(device)

    logger.print(f"Teacher model successfully loaded")

    # pass every necessary parameter to a models training via the model config

    teacher_config["epochs"] = EPOCHS
    teacher_config["learning_rate"] = LEARNING_RATE

    if not trained:
        train_model(teacher_model, train_loader, teacher_config, device, logger)
        if teacher_config["save_model"]:
            save_model(teacher_model, teacher_config)
        
    teacher_acc = test_model(teacher_model, test_loader, device, logger)

    logger.print(f"Teacher model accuracy: {teacher_acc}")

    # ablation study on temperature
    temperature = list(range(20, 52, 2))
    accuracy = []
    student_config = load_config(config["student_model_config"])

    student_config["epochs"] = EPOCHS
    student_config["learning_rate"] = LEARNING_RATE
    student_config["distillation_weight"] = config["distillation_weight"]
    student_config["ce_weight"] = config["ce_weight"]

    # lets pass every nece

    record = {"temperature": [], "seed": [], "accuracy": []}
    for seed in range(2):
        for temp in temperature:

            logger.print(f"Training student model with temperature: {temp}")

            student_config["temperature"] = temp
            student_model, trained = load_model(student_config, device, PARALLEL)

            student_model = train_distillation_model(teacher_model, student_model, train_loader, device, logger, student_config)
            student_acc = test_model(student_model, test_loader, device, logger)

            accuracy.append(student_acc)

            record["temperature"].append(temp)
            record["seed"].append(seed)
            record["accuracy"].append(student_acc)

        logger.print(f"Accuracy for temperature: {temperature} is {accuracy}")

    record = pd.DataFrame(record)
    record.to_csv("temperature_ablation.csv", index=False)


    # assert(0)

    # student_config = load_config(config["student_model_config"])

    # # teacher_model = nn.DataParallel(teacher_model)#.to(device)

    # student_model1 = load_model(student_config, device, PARALLEL)
    # student_model2 = load_model(student_config, device, PARALLEL)
    
    # ## to make the students weight identical
    # student_model2.load_state_dict(student_model1.state_dict())

    # # assert torch.norm(student_model1.module.conv_blocks.0.conv.weight - student_model2.module.conv_blocks.0.conv.weight) == 0
    # logger.print(f"Student model successfully loaded")

    # """Training the models"""

    # logger.print("Training the models")

    # logger.print(f"Training student 1 model: {student_config['model']}")    
    # # student_model1 = train_model(student_model1, train_loader, EPOCHS, device, logger)
    # # student_acc1 = test_model(student_model1, test_loader, device, logger)

    # logger.print(f"Training student 2 model: {student_config['model']}")
    # student_model2 = train_distillation_model(teacher_model, student_model2, train_loader, device, logger, config)
    # student_acc2 = test_model(student_model2, test_loader, device, logger) 

    # # if not teacher_config["load_from_path"]:
    # #     teacher_model = train_model(teacher_model, trainloader, EPOCHS, device, logger)

    # #     if teacher_config["save_path"]:
    # #         torch.save(teacher_model.state_dict(), teacher_config["save_path"])
    # #         logger.print(f"Teacher model saved at {teacher_config['save_path']}")



    

        
    



















if __name__ == "__main__":
    fire.Fire(main)