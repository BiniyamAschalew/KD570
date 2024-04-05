import torch.nn as nn
import torch

# creating convnet builder for MNIST dataset, with pooling
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
    
    
class ConvNetBuilder(nn.Module):

    def __init__(self, input_channels, output_channels, input_dict):

        super(ConvNetBuilder, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.input_dict = input_dict
        self.num_layers = len(self.input_dict["size"])

        self.conv_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for layer in range(self.num_layers):
            if layer == 0:
                in_channels = self.input_channels
            else:
                in_channels = self.input_dict["size"][layer-1]

            out_channels = self.input_dict["size"][layer]
            kernel_size = self.input_dict["kernel_size"][layer]

            stride = self.input_dict["stride"][layer]
            padding = self.input_dict["padding"][layer]
            dropout = self.input_dict["dropout"][layer]

            self.conv_blocks.append(ConvBlock(in_channels, out_channels, kernel_size, stride, padding, dropout))

        self.fc = nn.Linear(self.input_dict["size"][-1], self.output_channels)

    def forward(self, x):
        for layer in range(self.num_layers):
            x = self.conv_blocks[layer](x)
            x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    




# test
input_dict = { "size": [32, 64, 128, 256], "kernel_size": [3, 3, 3, 3], "stride": [1, 1, 1, 1], "padding": [1, 1, 1, 1], "dropout": [0.5, 0.5, 0.5, 0.5] }
input_channels = 1
output_channels = 10

convnet4 = ConvNetBuilder(input_channels, output_channels, input_dict)

print("successfully created convnet4 for MNIST dataset with pooling")
print(convnet4)

# test: creating a random tensor and forwarding it through the network

x = torch.randn(1, 1, 28, 28)
output = convnet4(x)

print("successfully forwarded a random tensor through the network")
print(output.shape)