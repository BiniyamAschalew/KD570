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

    # ablation study on transfer dataset
    transfer_dataset = ["SVHN", "SVHN/ActMax", "SVHN+ActMax"]
    accuracy = []
    student_config = load_config(config["student_model_config"])

    student_config["epochs"] = EPOCHS
    student_config["temperature"] = config["temperature"]
    student_config["learning_rate"] = LEARNING_RATE
    # student_config["distillation_weight"] = config["distillation_weight"]
    # student_config["ce_weight"] = config["ce_weight"]

    # lets pass every nece

    record = {"transfer_dataset": [], "trial0": [], "trial1": [], "trial2": [], "trial3": [], "trial4": [], "average": []}
    
    for dataset in transfer_dataset:
        if dataset == "None":
            student_config["distillation_weight"] = config["distillation_weight"]
            student_config["ce_weight"] = config["ce_weight"]
            
        elif "-" not in dataset:
            train_loader, test_loader = load_dataset(dataset, config["batch_size"])
            student_config["distillation_weight"] = config["distillation_weight"]
            student_config["ce_weight"] = config["ce_weight"]
            
        else:
            train_loader, test_loader = load_dataset(dataset.split("-")[0], config["batch_size"])
            
            if dataset.split("-")[1] == 'hard':
                student_config["distillation_weight"] = 0
                student_config["ce_weight"] = 1
                
            elif dataset.split("-")[1] == 'vanilla':
                student_config["distillation_weight"] = 0.8
                student_config["ce_weight"] = 0.2
                
            else:
                raise NotImplementedError
            
        logger.print(f"Training student model with transfer dataset {dataset}")
        record["transfer_dataset"].append(dataset)
        accuracies = []
        
        for trial in range(5):
            student_config["seed"] = trial * SEED
            student_model, trained = load_model(student_config, device, PARALLEL)
            if dataset != "None":
                student_model = train_distillation_model(teacher_model, student_model, train_loader, device, logger, student_config)
            student_acc = test_model(student_model, test_loader, device, logger, student_config)
            record[f"trial{trial}"].append(student_acc)
            accuracies.append(student_acc)

        accuracy.append(sum(accuracies)/5.0)
        record["average"].append(accuracy[-1])
        logger.print(f"Accuracy for transfer dataset {dataset} is {accuracy[-1]} (trials: {accuracies})")

    record = pd.DataFrame(record)
    logger.log_dataframe(record)

if __name__ == "__main__":
    fire.Fire(main)
