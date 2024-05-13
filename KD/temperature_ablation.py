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
    teacher_config["seed"] = config["seed"]

    if not trained:
        train_model(teacher_model, train_loader, teacher_config, device, logger)
        if teacher_config["save_model"]:
            save_model(teacher_model, teacher_config)
        
    teacher_acc = test_model(teacher_model, test_loader, device, logger)

    logger.print(f"Teacher model accuracy: {teacher_acc}")

    # ablation study on temperature
    temperature = [1] + list(range(10, 110, 10))   # [1, 10, 20, 30, ..., 100]
    accuracy = []
    student_config = load_config(config["student_model_config"])


    student_config["epochs"] = EPOCHS
    student_config["learning_rate"] = LEARNING_RATE
    student_config["distillation_weight"] = config["distillation_weight"]
    student_config["ce_weight"] = config["ce_weight"]

    # lets pass every nece

    record = {"temperature": [], "seed": [], "accuracy": []}
    for seed in range(5):
        for temp in temperature:

            logger.print(f"Training student model with temperature: {temp}")

            student_config["seed"] = seed
            student_config["temperature"] = temp
            student_model, trained = load_model(student_config, device, PARALLEL)

            student_model = train_distillation_model(teacher_model, student_model, train_loader, device, logger, student_config)
            student_acc = test_model(student_model, test_loader, device, logger)

            accuracy.append(student_acc)

            record["temperature"].append(temp)
            record["seed"].append(seed)
            record["accuracy"].append(student_acc)

        logger.print(f"Accuracy for temperature {temperature} is {accuracy} (seed:{seed})")

    record = pd.DataFrame(record)
    logger.log_dataframe(record)

if __name__ == "__main__":
    fire.Fire(main)
