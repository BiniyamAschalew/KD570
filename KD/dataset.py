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
        print("Loaded MNIST dataset")

    elif dataset.lower() == "cifar10":

        cifar10 = CIFAR10DataLoader(batch_size)
        train_loader, test_loader = cifar10.load_data()
        print("Loaded CIFAR10 dataset")


    elif dataset.lower() == "noise":
            
        noise = NoiseDataLoader(batch_size, size)
        train_loader, test_loader = noise.load_data()
        print("Loaded Noise dataset")


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

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet values
        ])

        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloader, test_loader
    

class NoiseDataLoader(KDataLoader):
    def __init__(self, batch_size: int, shape: List[int]):
        super(NoiseDataLoader, self).__init__(batch_size)
        self.shape = shape

    def load_data(self) -> Tuple[DataLoader, DataLoader]:

        train_data = torch.randn(self.shape)
        test_data = torch.randn(self.shape)

        trainset = TensorDataset(train_data)
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        testset = TensorDataset(test_data)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
    


    
    

