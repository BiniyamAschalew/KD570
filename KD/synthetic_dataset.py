import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from models import load_model
from utils import load_config

class SyntheticGenerator():
    def __init__(self, pretrained_model, data_config, device, logger, size_per_class, iterations=1000):
        self.model = pretrained_model
        self.size = size_per_class
        self.device = device
        self.logger = logger
        self.iterations = iterations
        self.data_config = data_config

        self.model.eval()
        self.model.to(self.device)

        # Transformations, if necessary, adjust according to actual need
        self.transform = transforms.Compose([
            transforms.Resize(self.data_config["input_shape"][1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust as per actual dataset mean/std
        ])

    def generate_dataset(self, batch_size=64, output_dir=None):
        data = []
        data_labels = []

        for label in range(self.data_config["num_classes"]):
            for _ in range(self.size):
                gen_img = self.generate_batch(label, batch_size)
                data.append(gen_img)
                data_labels.append(torch.full((batch_size,), label, dtype=torch.long))

        data = torch.cat(data)
        data_labels = torch.cat(data_labels)

        dataloader = DataLoader(TensorDataset(data, data_labels), batch_size=batch_size, shuffle=True)

        if output_dir:
            print("Saving dataset to ", output_dir)
            torch.save(dataloader, output_dir)

        return dataloader

    def generate_batch(self, label, batch_size):
        image = self.random_image(batch_size).to(self.device)
        image.requires_grad = True
        optimizer = optim.Adam([image], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

        for _ in range(self.iterations):
            optimizer.zero_grad()
            output = self.model(image)
            class_loss = -output[:, label].sum()

            # Regularization terms
            l2_loss = 0.001 * torch.norm(image)
            tv_loss = 0.001 * (torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) +
                               torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])))
            
            loss = class_loss + l2_loss + tv_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                image.clamp_(0, 1)  # Keep image values within valid range

        return image.detach()

    def random_image(self, batch_size=64):
        # Initialize close to the dataset mean
        random_image = torch.rand(batch_size, *self.data_config["input_shape"], device=self.device)
        random_image = random_image * 0.5 + 0.5  # Rescale to [0.5, 1.0]
        return random_image

def visualize_image(input_image: torch.Tensor):
    """given a torch tensor of shape [1, 28, 28] visualize an image of the MNIST dataset"""
    input_image = input_image.cpu().detach().numpy().squeeze()
    plt.imshow(input_image, cmap='gray')
    plt.show()


