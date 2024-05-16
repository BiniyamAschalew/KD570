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
    SEED = config["seed"]

    set_seed(config["seed"])

    train_loader, test_loader = load_dataset(config["dataset"], config["batch_size"])

    teacher_config = load_config(config["teacher_model_config"])
    teacher_model, trained = load_model(teacher_config, device, PARALLEL)
    # teacher_model = nn.DataParallel(teacher_model).to(device)

    logger.print(f"Teacher model successfully loaded")

    # pass every necessary parameter to a models training via the model config

    teacher_config["epochs"] = EPOCHS
    teacher_config["learning_rate"] = LEARNING_RATE
    teacher_config["seed"] = SEED

    if not trained:
        train_model(teacher_model, train_loader, teacher_config, device, logger)
        if teacher_config["save_model"]:
            save_model(teacher_model, teacher_config)
        
    teacher_acc = test_model(teacher_model, test_loader, device, logger, teacher_config)

    logger.print(f"Teacher model accuracy: {teacher_acc}")

    # ablation study on learning rate
    learning_rate = [0.0000625, 0.000125, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]
    accuracy = []
    student_config = load_config(config["student_model_config"])

    student_config["epochs"] = EPOCHS
    student_config["temperature"] = config["temperature"]
    student_config["distillation_weight"] = config["distillation_weight"]
    student_config["ce_weight"] = config["ce_weight"]

    # lets pass every nece

    record = {"learning_rate": [], "trial0": [], "trial1": [], "trial2": [], "trial3": [], "trial4": [], "average": []}
    
    for lr in learning_rate:
        student_config["learning_rate"] = lr
        logger.print(f"Training student model with learning rate {lr}")
        record["learning_rate"].append(lr)
        accuracies = []
        
        for trial in range(5):
            student_config["seed"] = trial * SEED
            student_model, trained = load_model(student_config, device, PARALLEL)
            student_model = train_distillation_model(teacher_model, student_model, train_loader, device, logger, student_config)
            student_acc = test_model(student_model, test_loader, device, logger, student_config)
            record[f"trial{trial}"].append(student_acc)
            accuracies.append(student_acc)

        accuracy.append(sum(accuracies)/5.0)
        record["average"].append(accuracy[-1])
        logger.print(f"Accuracy for learning rate {lr} is {accuracy[-1]} (trials: {accuracies})")

    record = pd.DataFrame(record)
    logger.log_dataframe(record)

if __name__ == "__main__":
    fire.Fire(main)
