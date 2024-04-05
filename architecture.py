import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


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
        

def train_network(model, trainloader, epochs, lr):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print("Finished Training")
    return model

def test_network(model, testloader):

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return correct / total

def get_softmax_output(model, data):
    images, labels = data
    outputs = model(images)
    softmax = nn.Softmax(dim=1)
    softmax_outputs = softmax(outputs)
    return softmax_outputs

