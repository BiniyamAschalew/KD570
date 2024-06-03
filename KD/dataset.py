import typing
from typing import Tuple, List
from abc import abstractmethod
import numpy as np
import os, pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def combine_dataloaders(synthetic_loader: torch.utils.data.DataLoader, real_loader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    
    def get_data_and_label(dataloader):
        all_images = []
        all_labels = []
        
        for images, labels in dataloader:
            all_images.append(images)
            all_labels.append(labels)

        all_images = torch.cat(all_images)
        all_labels = torch.cat(all_labels)
        
        return all_images, all_labels

    synthetic_data, synthetic_label = get_data_and_label(synthetic_loader)
    real_data, real_label = get_data_and_label(real_loader)
    
    data = torch.cat([synthetic_data, real_data])
    label = torch.cat([synthetic_label, real_label])
    
    combined_dataset = torch.utils.data.TensorDataset(data, label)
    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=synthetic_loader.batch_size, shuffle=True)
    
    return combined_loader
    
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
        
    elif dataset.lower() == "svhn":
        svhn = SVHNDataLoader(batch_size)
        train_loader, test_loader = svhn.load_data()
        print("Loaded SVHN dataset")
        
    elif dataset.lower() == "imagenet32":
        imagenet = ImageNetDataLoader(batch_size, image_size=32)
        train_loader, test_loader = imagenet.load_data()
        print("Loaded ImageNet32 dataset")

    elif "/" in dataset.lower():    # assumes input like MNIST/CycleGAN
        model = dataset.split("/")[1]
        dataset = dataset.split("/")[0]
        synthetic = SyntheticDataLoader(dataset, model, batch_size)
        train_loader, test_loader = synthetic.load_data()
        print(f"Loaded synthetic {dataset} dataset generated using {model}")

    elif "+" in dataset.lower():    # assumes input like MNIST+CycleGAN
        model = dataset.split("+")[1]
        dataset = dataset.split("+")[0]
        synthetic = SyntheticDataLoader(dataset, model, batch_size, test_loader=False)
        synthetic_loader, _ = synthetic.load_data()

        if dataset.lower() == "mnist":
            mnist = MNISTDataLoader(batch_size)
            real_loader, test_loader = mnist.load_data()

        elif dataset.lower() == "svhn":
            svhn = SVHNDataLoader(batch_size)
            real_loader, test_loader = svhn.load_data()
        
        elif dataset.lower() == "cifar10":
            cifar10 = CIFAR10DataLoader(batch_size)
            real_loader, test_loader = cifar10.load_data()

        elif dataset.lower() == "imagenet32":
            imagenet = ImageNetDataLoader(batch_size, image_size=32)
            real_loader, test_loader = imagenet.load_data()

        else:
            raise NotImplementedError

        train_loader = combine_dataloaders(synthetic_loader, real_loader)
        print(f"Loaded real and synthetic {dataset} dataset generated using {model}")

    return train_loader, test_loader


class KDataLoader(nn.Module):
    """Abstract class for data loaders, all data loaders should follow this strucutre"""

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
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  
        #     transforms.ToTensor(), 
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet values
        #     transforms.Resize((32, 32))
        # ])

        trainset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        test_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)
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
    

class SVHNDataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load_data(self): # don't donwload if already downloaded
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.SVHN(root='./data/SVHN', split='train', download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        testset = datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return trainloader, testloader
    
    
class ImageNetDataLoader:
    def __init__(self, batch_size, image_size=32):
        self.batch_size = batch_size
        self.image_size = image_size

    def load_data(self):
        trainbatch = [self._load_databatch(f'./data/ImageNet{self.image_size}/train', i) for i in range(1,11)]
        trainset = TensorDataset(torch.cat([batch['image'] for batch in trainbatch]), torch.cat([batch['label'] for batch in trainbatch]))
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        testbatch = [self._load_databatch(f'./data/ImageNet{self.image_size}/val', None)]
        testset = TensorDataset(torch.cat([batch['image'] for batch in testbatch]), torch.cat([batch['label'] for batch in testbatch]))
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        
        return trainloader, testloader
    
    # Note that this will work with Python3
    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def _load_databatch(self, data_folder, idx):
        if idx is not None:
            data_file = os.path.join(data_folder, f'train_data_batch_{idx}')
        else:
            data_file = os.path.join(data_folder, 'val_data')

        d = self._unpickle(data_file)
        x = d['data']/float(255)     # change to 0. to 1.
        x = (x - 0.5) * 2            # normalize to -1. to +1.
        y = d['labels']

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]
        data_size = x.shape[0]

        img_size2 = self.image_size * self.image_size

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], self.image_size, self.image_size, 3)).transpose(0, 3, 1, 2)

        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        Y_train = y[0:data_size]
        X_train_flip = X_train[:, :, :, ::-1]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train, X_train_flip), axis=0)
        Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)
        
        # only consider cases where label is 0-49 (50 classes)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.uint8)
        mask = (Y_train >= 0) & (Y_train <= 49)
        
        return dict(
            image=X_train[mask], # float from -1. to +1.
            label=Y_train[mask]) 
        
        
class SyntheticDataLoader:
    def __init__(self, dataset, model, batch_size, test_loader=True):
        self.batch_size = batch_size
        self.test_loader = test_loader
        self.end = 120
        
        if dataset.lower().strip() == 'imagenet32':
            self.dataset = 'ImageNet32'
            self.end = 5
        elif dataset.lower().strip() == 'mnist':
            self.dataset = 'MNIST'
            self.end = 120
        elif dataset.lower().strip() == 'cifar10':
            self.dataset = 'CIFAR10'
            self.end = 100
        elif dataset.lower().strip() == 'svhn':
            self.dataset = 'SVHN'
            self.end = 150
        else:
            raise NotImplementedError
        
        if model.lower().strip() == 'ddpm':
            self.model = 'DDPM'
        elif model.lower().strip() == 'cyclegan':
            self.model = 'CycleGAN'
            if self.dataset == 'SVHN':
                self.end = 120
        elif model.lower().strip() == 'noise':
            self.model = 'Noise'
        elif model.lower().strip() == 'actmax':
            self.model = 'ActMax'
        elif model.lower().strip() == 'cgan':
            self.model = 'cGAN'
        elif model.lower().strip() == 'pix2pix':
            self.model = 'pix2pix'
        else:
            raise NotImplementedError

    def load_data(self):
        trainbatch = [self._load_databatch(f'./data/synthetic/{self.dataset}/{self.model}/{i}_of_{self.end}.npz') for i in range(1,(self.end+1))]
        trainset = TensorDataset(torch.cat([batch['image'] for batch in trainbatch]), torch.cat([batch['label'] for batch in trainbatch]))
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        # test split is still using real data
        if self.test_loader:
            if self.dataset == 'MNIST':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])    
                testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
                testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
                
            elif self.dataset == 'SVHN':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                testset = datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform)
                testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
            
            elif self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                testset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)
                testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
            
            elif self.dataset == 'ImageNet32':
                self.image_size = 32
                testbatch = [self._load_databatch_imagenet(f'./data/ImageNet{self.image_size}/val', None)]
                testset = TensorDataset(torch.cat([batch['image'] for batch in testbatch]), torch.cat([batch['label'] for batch in testbatch]))
                testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
            
            else:
                raise NotImplementedError
        
        else:
            testloader = None

        return trainloader, testloader

    def _load_databatch(self, path):
        loaded_data = np.load(path)
        try:
            loaded_tensor_cpu = loaded_data['data']
        except:
            loaded_tensor_cpu = loaded_data['arr_0']
        X_train = torch.tensor(loaded_tensor_cpu)
        Y_train = torch.zeros(X_train.size(0))
        
        return dict(
            image=X_train.to(torch.float32), # float from -1.0 to 1.0
            label=Y_train.to(torch.uint8))   # tensor of zeros

    # Note that this will work with Python3
    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict
        
    def _load_databatch_imagenet(self, data_folder, idx):
        if idx is not None:
            data_file = os.path.join(data_folder, f'train_data_batch_{idx}')
        else:
            data_file = os.path.join(data_folder, 'val_data')

        d = self._unpickle(data_file)
        x = d['data']/float(255)
        x = (x - 0.5) * 2
        y = d['labels']

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]
        data_size = x.shape[0]

        img_size2 = self.image_size * self.image_size

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], self.image_size, self.image_size, 3)).transpose(0, 3, 1, 2)

        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        Y_train = y[0:data_size]
        X_train_flip = X_train[:, :, :, ::-1]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train, X_train_flip), axis=0)
        Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)
        
        # only consider cases where label is 0-49 (50 classes)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.uint8)
        mask = (Y_train >= 0) & (Y_train <= 49)
        
        return dict(
            image=X_train[mask], # float from -1. to +1.
            label=Y_train[mask])
