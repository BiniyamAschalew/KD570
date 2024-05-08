import typing
from typing import Tuple, List
from abc import abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def load_dataset(dataset, batch_size, size = None):
    
    if dataset.lower() == "mnist":

        mnist = MNISTDataLoader(batch_size)
        train_loader, test_loader = mnist.load_data()

    elif dataset.lower() == "cifar10":

        cifar10 = CIFAR10DataLoader(batch_size)
        train_loader, test_loader = cifar10.load_data()

    elif dataset.lower() == "noise":
            
        noise = NoiseDataLoader(batch_size, size)
        train_loader, test_loader = noise.load_data()

    return train_loader, test_loader


class KDataLoader(nn.Module):
    """bstract class for data loaders, all data loaders should follow this strucutre"""

    def __init__(self, batch_size: int):
        super(KDataLoader, self).__init__()
        self.batch_size = batch_size

    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        pass
        

class MNISTDataLoader(KDataLoader):
    def __init__(self, batch_size: int):
        super(MNISTDataLoader, self).__init__(batch_size)

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloader, test_loader
    

class CIFAR10DataLoader(KDataLoader):
    def __init__(self, batch_size: int):
        super(CIFAR10DataLoader, self).__init__(batch_size)

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloader, test_loader
    

class NoiseDataLoader(KDataLoader):
    # if it is not saved then generate and save it for later use
    def __init__(self, batch_size: int, shape: List[int]):
        super(NoiseDataLoader, self).__init__(batch_size)
        self.shape = shape

    def load_data(self) -> Tuple[DataLoader, DataLoader]:

        pass


    pass

    

