# generating synthetic dataset using activation maximization for each output class of the given model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from PIL import Image

from models import load_model
from utils import load_config


class SyntheticGenerator():
    def __init__(self, pretrained_model, data_config, device, logger, size_per_class):

        self.model = pretrained_model
        self.size = size_per_class
        self.device = device
        self.logger = logger

        self.data_config = data_config

        print(self.data_config["input_shape"])
        self.transform = transforms.Compose([
            transforms.Resize(self.data_config["input_shape"][1:]),
            transforms.ToTensor()
        ])

        self.model.eval()
        self.model.to(self.device)

    def generate_dataset(self, output_dir = None):

        data = []
        data_labels = []
        dataloader = None

        for _ in range(self.size):
            for label in range(self.data_config["num_classes"]):

                gen_img = self.generate_image(label, output_dir)
                data.append(gen_img)
                data_labels.append(label)

        data = torch.stack(data)
        data_labels = torch.tensor(data_labels)

        dataloader = DataLoader(TensorDataset(data, data_labels), batch_size=32, shuffle=True)

        if output_dir:
            print("Saving dataset to ", output_dir)
            torch.save(dataloader, output_dir)

        return dataloader


    def generate_image(label):

        image = self.random_image().to(self.device)
        image.requires_grad = True

        optimizer = optim.Adam([image], lr=0.01)

        for _ in range(1000):

            optimizer.zero_grad()
            output = self.model(image)
            loss = -output[0][label]
            loss.backward()
            optimizer.step()

        return image


    def random_image(self):

        random_high, random_low = 180, 160

        mean=torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        std=torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        random_high, random_low = 180, 160
        
        image = (((random_high - random_low) * torch.rand(self.data_config["input_shape"]) + random_low)/255)
        image = (image - mean) / std

        return image


if __name__ == "__main__":

    config_dir = "configs/model_configs/ResNet18_MNIST.yaml"
    model_config = load_config(config_dir)
    model, trained = load_model(model_config, torch.device("cuda"))

    if not trained:
        raise ValueError("Model must be trained.")

    dataset_dir = "configs/dataset_configs/MNIST.yaml"
    data_config = load_config(dataset_dir)

    device = torch.device("cuda")
    logger = None
    size_per_class = 1

    generator = SyntheticGenerator(model, data_config, device, logger, size_per_class)


    print("model loaded")
