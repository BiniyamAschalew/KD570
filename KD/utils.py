import datetime
import torch
# import logging

class Logger:

    def __init__(self, log_dir):
        self.log_dir = log_dir

    def store(self, message):

        current_time = datetime.datetime.now().strfttime("%%d %H:%M")
        file_name = self.log_dir + f"/log{current_time}.txt"

        with open(file_name, "w") as file:
            file.write(message)

def load_config(config_dir):
    with open(config_dir, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_logger(log_dir):
    logger = Logger(log_dir)
    return logger

