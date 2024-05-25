import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import numpy as np
import matplotlib.pyplot as plt
import fire 
from tqdm import tqdm

from models import load_model
from utils import load_config, get_logger


class SyntheticGenerator():
    def __init__(self, pretrained_model, data_config, device, logger, size, iterations=1000):

        self.model = pretrained_model
        self.size = size
        self.device = device

        self.logger = logger
        self.iterations = iterations
        self.data_config = data_config

        self.model.eval()

        # should use 
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        
        self.model.to(self.device)

        # self.size_per_label = self.size // self.data_config["num_classes"]

        t_mean = self.data_config["mean"]
        t_std = self.data_config["std"]

        self.transform = transforms.Compose([
            transforms.Resize(self.data_config["input_shape"][1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean = t_mean, std = t_std)
        ])


    def generate_dataset(self):

        data = []
        data_labels = []
        num_labels = self.data_config["num_classes"]
        batch_size = self.size // num_labels

        self.logger.print(f"Generator instantiated \n In this iteration generator is required to generate {self.size} number of images")
        self.logger.print(f"generating {batch_size} images for each of the {num_labels} labels")
        
        for label in tqdm(range(num_labels)):
            self.logger.print(f"Generating images for label {label}")


            image_batch = self.generate_batch(label, batch_size)
            data.append(image_batch)
            data_labels.append(torch.full((batch_size,), label, dtype=torch.long))

        data = torch.cat(data)
        data_labels = torch.cat(data_labels)

        print(f"The shape of the generated dataset is {data.shape} and labels is {data_labels.shape}")

        dataloader = DataLoader(TensorDataset(data, data_labels), shuffle=True)
        return dataloader
    

    def generate_batch(self, label, batch_size):

        image = self.random_image(batch_size).to(self.device)
        image.requires_grad = True
        optimizer = optim.Adam([image], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

        self.logger.print(f"Generating a batch of {batch_size} images")
        for _ in tqdm(range(self.iterations)):
            optimizer.zero_grad()
            output = self.model(image)
            class_loss = -output[:, label].sum()

            # regularization to make the images more visisble
            l2_loss = 0.001 * torch.norm(image)
            tv_loss = 0.001 * (torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) +
                               torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])))
            
            loss = class_loss + l2_loss + tv_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                image.clamp_(0, 1)  # gradient clipping for stable generation

        return image.detach()

    def random_image(self, batch_size):
        # setting the initial value of the random image to be close to the dataset distribution
        random_image = torch.randn(batch_size, *self.data_config["input_shape"], device=self.device)
        
        mean = torch.tensor(self.data_config["mean"]).view(1, -1, 1, 1).to(self.device)
        std = torch.tensor(self.data_config["std"]).view(1, -1, 1, 1).to(self.device)

        random_image = random_image * std + mean
        return random_image



def visualize_image(input_image: torch.Tensor):
    input_image = input_image.cpu().detach().numpy()

    num_channels = input_image.shape[0]
    if num_channels == 1:
        plt.imshow(np.transpose(input_image, (1, 2, 0))[:, :, 0], cmap="gray")

    elif num_channels == 3:
        plt.imshow(np.transpose(input_image, (1, 2, 0)))

    else: 
        raise ValueError("Input image must have 1 or 3 channels.")

    plt.show()


def generate( model_config: str, dataset_config: str, size: int, total_size: int, iterations: int, device: str = "cuda"):


    model_config = load_config(model_config)
    dataset_config = load_config(dataset_config)
    device = torch.device(device)
    logger = get_logger()

    name = dataset_config["dataset"]

    logger.print("Loading model")
    model, trained = load_model(model_config, torch.device("cuda"))
    logger.print(f"Model {model_config['model']} loaded, with trained status {trained}")
    
    if not trained:
        raise ValueError("Model must be trained.")

    print(f"Generating {total_size} images\n\n\n")
    num_sampling = total_size // size #120
    generator = SyntheticGenerator(model, dataset_config, device, logger, size, iterations)

    # the generator generates {size} number of images resulting in total of {size} * {num_sampling} = {total_size} images
    data_loader = generate_save(generator, num_sampling, name)

    return data_loader


def generate_save(generator, num_sampling, name):

    # save_dir = "./data/synthetic/MNIST/ACTIVATION/"
    save_dir = f"./data/synthetic/{name}/ACTIVATION/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_sampling):

        data_loader = generator.generate_dataset()

        synthetic_data, synthetic_label = data_loader.dataset.tensors
        # x_all = torch.cat([x_gen])
        synthetic_data = torch.clamp(synthetic_data, min=-1.00, max=1.00)

        # np.savez(f'./data/synthetic/MNIST/ACTIVATION/{i+1}_of_{num_sampling}.npz', data=x_gen.cpu().detach().numpy())
        np.savez(f'{save_dir}{i+1}_of_{num_sampling}.npz', data=synthetic_data.cpu().detach().numpy())

        # randomly select 25 to viusalize
        random_sub_x_gen = synthetic_data[np.random.choice(synthetic_data.shape[0], 5, replace=False)]

        grid = make_grid(random_sub_x_gen*-1 + 1, nrow=5)
        # save_image(grid,  f"./data/synthetic/MNIST/ACTIVATION/{i+1}_of_{num_sampling}.png")
        save_image(grid,  f"{save_dir}{i+1}_of_{num_sampling}.png")

        # print('saved image at ' + f"./data/synthetic/MNIST/ACTIVATION/{i+1}_of_{num_sampling}.npz")
        print('saved image at ' + f"{save_dir}{i+1}_of_{num_sampling}.npz")

    return data_loader



def main(config_dir: str):

    config = load_config(config_dir)

    

    data_loader = generate( model_config= config["model_config"],
                            dataset_config=config["dataset_config"],
                            size=config["size"],
                            total_size = config["total_size"],
                            iterations = config["iterations"],
                            device=config["device"],
                            )
    

    # vis = 10
    # for i in range(vis):
    #     image, label = next(iter(data_loader))
    #     print("Label: ", label[0].item(), "Shape: ", image[0].shape)
    #     visualize_image(image[0])
        
    return data_loader


if __name__ == "__main__":
    fire.Fire(main)
