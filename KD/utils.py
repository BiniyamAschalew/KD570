import datetime
import torch
import numpy 
import random
import yaml
import pandas as pd
# import logging

class Logger:

    def __init__(self, log_dir):
        self.log_dir = log_dir
        current_time = datetime.datetime.now().strftime("%d %H:%M")
        self.log_dir = self.log_dir + f"/log{current_time}.txt"

        with open(self.log_dir, "w") as file:
            file.write("Begin logging\n\n")


    def print(self, message):

        message = "\n" + message + "\n"
        with open(self.log_dir, "a") as file:
            file.write(message)
        
        print(message)

    def log_dataframe(self, dataframe):

        dataframe.to_csv(self.log_dir.replace('.txt', '.csv'), index=False)
        self.print(f"Dataframe logged at {self.log_dir.replace('.txt', '.csv')}")
        


def load_config(config_dir):
    with open(config_dir, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config



def get_device(device: str) -> torch.device:
    available_device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != available_device:
        raise ValueError(f"Device {device} not available.")

    device = torch.device(device)
    return device

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_logger(log_dir = "./logs"):
    logger = Logger(log_dir)
    return logger

def set_seed(seed):

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
