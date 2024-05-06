import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.5):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
# lets assume the input is (1, 10, {"size": [16, 32], "kernel_size": [5, 5], "stride": [1, 1], "padding": [2, 2], "dropout": [0.5, 0.5]})

class ConvNetBuilder(nn.Module):
    def __init__(self, input_channels, output_channels, input_dict, input_size=28): 
        super(ConvNetBuilder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_dict = input_dict
        self.num_layers = len(self.input_dict["size"])
        self.conv_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        current_size = input_size
        for layer in range(self.num_layers):
            in_channels = self.input_channels if layer == 0 else self.input_dict["size"][layer-1]
            out_channels = self.input_dict["size"][layer]
            kernel_size = self.input_dict["kernel_size"][layer]
            stride = self.input_dict["stride"][layer]
            padding = self.input_dict["padding"][layer]
            dropout = self.input_dict["dropout"][layer]

            self.conv_blocks.append(ConvBlock(in_channels, out_channels, kernel_size, stride, padding, dropout))

            current_size = (current_size + 2 * padding - kernel_size) // stride + 1
            current_size = current_size // 2

        flattened_size = current_size * current_size * self.input_dict["size"][-1]
        self.fc = nn.Linear(flattened_size, self.output_channels)

    def forward(self, x):
        for layer in range(self.num_layers):
            x = self.conv_blocks[layer](x)
            x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
        
