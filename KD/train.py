from typing import List, Tuple
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def save_model(model: nn.Module, model_config: dict): 

    if model_config["save_model"]:
        torch.save(model.state_dict(), model_config["save_path"])
        print(f"Model saved at {model_config['save_path']}")
    else:
        print("Model not saved as save_model is set to False")

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

            if i % 100 == 99:
                logger.print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / 100}')


        logger.print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(trainloader)}')

    logger.print("Finished Training")
    return model


def train_distillation_model(teacher: nn.Module, student: nn.Module, trainloader: DataLoader,
                     device: torch.device, logger: object, config: dict) -> nn.Module:
    
        teacher.eval()
        student.train()
        teacher.to(device)
        student.to(device)

        epochs = config['epochs']
        temperature = config['temperature']
        distillation_weight = config['distillation_weight']
        ce_weight = config['ce_weight']


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=0.001)
    
        for epoch in range(epochs):
            running_loss = 0.0
            for j,  (inputs, labels) in enumerate(trainloader):

    
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

                if j % 100 == 99:
                    logger.print(f"Epoch {epoch + 1}/{epochs} | Iteration {j + 1}/{len(trainloader)} | Loss: {running_loss / 100:.3f}")


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