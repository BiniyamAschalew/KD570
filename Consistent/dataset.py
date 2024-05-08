import typing
from typing import Tuple, List
from abc import abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

class KDataLoader(nn.Module):
    def __init__(self, batch_size: int):
        super(KDataLoader, self).__init__()
        self.batch_size = batch_size

    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        pass
        
class MNISTDataLoader(KDataLoader):
    def __init__(self, batch_size: int):
        super(MNISTDataLoader, self).__init__(batch_size)

    def load_data(self) -> Tuple[]