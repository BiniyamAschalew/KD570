from architecture import ConvNetBuilder
from utils import train_network, test_network, get_soft_labels, generate_noise_dataloader
from dataset import MNISTDataLoader
import torch
import random
import numpy as np

# setting a fixed seed
seed_value = 19
BATCH_SIZE = 32


torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


student_input_dict = { "size": [16, 32], "kernel_size": [3, 3], "stride": [1, 1], "padding": [0, 0], "dropout": [0, 0] }

student0 = ConvNetBuilder(1, 10, student_input_dict) # learns from softlabeled noise
student1 = ConvNetBuilder(1, 10, student_input_dict) # learns from hardlabeled mnist
student2 = ConvNetBuilder(1, 10, student_input_dict) # learns from softlabeled mnist
student3 = ConvNetBuilder(1, 10, student_input_dict) # no learning / random baseline

student0.to(device)
student1.to(device)
student2.to(device)
student3.to(device)

teacher_input_dict = { "size": [16, 32, 64, 64], "kernel_size": [3, 3, 3, 3], "stride": [1, 1, 1, 1], "padding": [1, 1, 1, 1], "dropout": [0.5, 0.5, 0.5, 0.5] }
teacher = ConvNetBuilder(1, 10, teacher_input_dict)

teacher.to(device)

trainloader, testloader = MNISTDataLoader(BATCH_SIZE).load_data()
teacher = train_network(teacher, trainloader, 1, 0.001, device)

print("finished training teacher")
test_network(teacher, testloader, device)

mnist_soft_labels_loader = get_soft_labels(teacher, trainloader, device)
print("finished getting soft labels for MNIST")

# mnist dataset has 60k images, for the noise we will generate 600k images
noise_loader = generate_noise_dataloader(100000, device, BATCH_SIZE)
noise_soft_labels_loader = get_soft_labels(teacher, noise_loader, device)
print("finished getting soft labels for noise")

student0 = train_network(student0, noise_soft_labels_loader, 1, 0.001, device)
# student1 = train_network(student1, trainloader, 1, 0.001, device)
student2 = train_network(student2, mnist_soft_labels_loader, 1, 0.001, device)

print("evaluating the four students")
print("evaluating student 0 which is trained on soft labels from noise")
test_network(student0, testloader, device)

print("evaluating student 1 which is trained on hard labels")
# test_network(student1, testloader, device)

print("evaluating student 2 which is trained on soft labels")
test_network(student2, testloader, device)

print("evaluating student 3 which is not trained yet")
test_network(student3, testloader, device)