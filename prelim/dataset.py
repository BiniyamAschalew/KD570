import torchvision
import torchvision.transforms as transforms
import torch


class MNISTDataLoader:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load_data(self): # don't donwload if already downloaded

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloader, test_loader