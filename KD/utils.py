import datetime
import torch
import numpy 
import random
import yaml
# import logging

class Logger:

    def __init__(self, log_dir):
        self.log_dir = log_dir


    def print(self, message):

        current_time = datetime.datetime.now().strfttime("%%d %H:%M")
        file_name = self.log_dir + f"/log{current_time}.txt"

        with open(file_name, "a") as file:
            file.write(message)
        
        print(message)

def load_config(config_dir):
    with open(config_dir, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def get_device(device: str) -> torch.device:
    available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != available_device:
        raise ValueError(f"Device {device} not available.")
    return device

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_logger(log_dir):
    logger = Logger(log_dir)
    return logger

def set_seed(seed):

    random.seed(seed)
    numpy.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    

