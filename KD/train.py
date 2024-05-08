from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_model(model: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device, logger: object):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    model.to(device)

    for epoch in range(epochs):

        epoch_loss = 0
        for i,  (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(trainloader)}')

    logger.print("Finished Training")
    return model


def train_distillation_model(teacher: nn.Module, student: nn.Module, trainloader: DataLoader, epochs: int,
                    temperature: float, distillation_weight: float, ce_weight: float, device: torch.device, logger: object) -> nn.Module:
    
        teacher.eval()
        student.train()
        teacher.to(device)
        student.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=0.001)
    
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in trainloader:
    
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
    
                student_logits = student(inputs)
                soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)

                # log softmax 
                student_log_soft = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    
                # the scaling by the T squared is suggested by the authors of "Distilling the Knowledge in a Neural Network"
                distillation_loss = torch.sum(soft_targets * (soft_targets.log() - student_log_soft)) / student_logits.size(0) * (temperature ** 2)
                ce_loss = criterion(student_logits, labels)
    
                total_loss = distillation_weight * distillation_loss + ce_weight * ce_loss
                total_loss.backward()
                optimizer.step()
    
                running_loss += total_loss.item()
            logger.print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {running_loss / len(trainloader):.3f}")
    
        return student 


def test_model(model: nn.Module, testloader: DataLoader, device: torch.device, logger: object) -> float:

    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    return accuracy