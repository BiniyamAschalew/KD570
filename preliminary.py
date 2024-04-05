from architecture import ConvNetBuilder
from utils import train_network, test_network, get_soft_labels
from dataset import MNISTDataLoader

student_input_dict = { "size": [16, 32], "kernel_size": [3, 3], "stride": [1, 1], "padding": [0, 0], "dropout": [0, 0] }
student1 = ConvNetBuilder(1, 10, student_input_dict)
student2 = ConvNetBuilder(1, 10, student_input_dict)
student3 = ConvNetBuilder(1, 10, student_input_dict)

teacher_input_dict = { "size": [16, 32, 64, 64], "kernel_size": [3, 3, 3, 3], "stride": [1, 1, 1, 1], "padding": [1, 1, 1, 1], "dropout": [0.5, 0.5, 0.5, 0.5] }
teacher = ConvNetBuilder(1, 10, teacher_input_dict)

trainloader, testloader = MNISTDataLoader(4).load_data()
teacher = train_network(teacher, trainloader, 1, 0.001)

print("finished training teacher")
test_network(teacher, testloader)

soft_labels_loader = get_soft_labels(teacher, trainloader)
print("finished getting soft labels")

student1 = train_network(student1, trainloader, 1, 0.001)
student2 = train_network(student2, soft_labels_loader, 1, 0.001)

print("evaluating the three students")
print("evaluating student 1 which is trained on hard labels")
test_network(student1, testloader)
print("evaluating student 2 which is trained on soft labels")
test_network(student2, testloader)
print("evaluating student 3 which is not trained yet")
test_network(student3, testloader)