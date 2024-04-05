import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def train_network(model, trainloader, epochs, lr, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

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

def test_network(model, testloader, device):

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:

            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return correct / total


def get_soft_labels(model, trainloader, device):
    model.eval()  # Set the model to evaluation mode
    soft_labels = []
    data_list = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():  # No need to track gradients for this operation
        for inputs, _ in trainloader:

            inputs = inputs.to(device)
            outputs = model(inputs)

            soft_labels.extend(softmax(outputs).tolist())
            data_list.extend(inputs)

    # Creating new dataset with the same data but with soft labels
    soft_labels_dataset = TensorDataset(torch.stack(data_list), torch.Tensor(soft_labels))
    soft_labels_loader = DataLoader(soft_labels_dataset, batch_size=trainloader.batch_size)

    return soft_labels_loader


def generate_noise_dataloader(size, device, batch_size=4):

    data = torch.randn(size, 1, 28, 28)
    data = (data - 0.5) / 0.5
    labels = torch.randint(0, 10, (size,))

    data, labels = data.to(device), labels.to(device)

    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def visualize_img(tensor):

    image_tenor = tensor * 0.5 + 0.5
    numpy_image = image_tenor.numpy()
    plt.imshow(numpy_image[0], cmap='gray')
    plt.show()
