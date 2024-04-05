from dataset import MNISTDataLoader
from architecture import ConvNetBuilder, train_network, test_network



input_dict = { "size": [16, 32, 64], "kernel_size": [3, 3, 3], "stride": [1, 1, 1], "padding": [1, 1, 1], "dropout": [0.5, 0.5, 0.5] }
model = ConvNetBuilder(1, 10, input_dict)
trainloader, testloader = MNISTDataLoader(4).load_data()
model = train_network(model, trainloader, 1, 0.001)

test_network(model, testloader)

teacher_input_dict = { "size": [16, 32, 64, 128], "kernel_size": [3, 3, 3, 3], "stride": [1, 1, 1, 1], "padding": [1, 1, 1, 1], "dropout": [0.5, 0.5, 0.5, 0.5] }
student_input_dict = { "size": [32, 32], "kernel_size": [3, 3,], "stride": [1, 1], "padding": [1, 1], "dropout": [0.5, 0.5] }

teacher = ConvNetBuilder(1, 10, teacher_input_dict)
student = ConvNetBuilder(1, 10, student_input_dict)

trainloader, testloader = MNISTDataLoader(4).load_data()
teacher = train_network(teacher, trainloader, 1, 0.001)
