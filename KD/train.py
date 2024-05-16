from typing import List, Tuple
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import set_seed

def save_model(model: nn.Module, model_config: dict): 

    if model_config["save_model"]:
        torch.save(model.state_dict(), model_config["save_path"])
        print(f"Model saved at {model_config['save_path']}")
    else:
        print("Model not saved as save_model is set to False")

def train_model(model: nn.Module, trainloader: DataLoader, model_config: dict, device: torch.device, logger: object):

    set_seed(model_config['seed'])

    logger.print(f"Starting training model {model_config['model']}")

    epochs = model_config['epochs']
    learning_rate = model_config['learning_rate']

    logger.print(f"Training with epochs: {epochs}, learning rate: {learning_rate}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    model.to(device)

    for epoch in range(epochs):

        running_loss = 0
        epoch_loss = 0
        for i,  (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f'epoch {epoch+1}/{epochs} | loss: {running_loss / 100}')
                running_loss = 0.0


        logger.print(f'epoch {epoch+1}/{epochs} | loss: {running_loss / len(trainloader)}')

    logger.print("Finished training")
    return model


def train_distillation_model(teacher: nn.Module, student: nn.Module, trainloader: DataLoader,
                     device: torch.device, logger: object, model_config: dict) -> nn.Module:
        
        logger.print(f"Starting distillation training model {model_config['model']}")
        set_seed(model_config['seed'])

        teacher.eval()
        student.train()
        teacher.to(device)
        student.to(device)

        epochs = model_config['epochs']
        temperature = model_config['temperature']
        learning_rate = model_config['learning_rate']
        distillation_weight = model_config['distillation_weight']
        ce_weight = model_config['ce_weight']

        logger.print(f"Distillation training with temperature: {temperature}, distillation weight: {distillation_weight}, ce weight: {ce_weight}, learning rate: {learning_rate}, seed: {model_config['seed']}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    
        for epoch in range(epochs):
            running_loss = 0.0
            for j,  (inputs, labels) in enumerate(trainloader):
    
                inputs, labels = inputs.to(device), labels.to(device).long()
                
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
                    print(f"epoch {epoch + 1}/{epochs} | iteration {j + 1}/{len(trainloader)} | loss: {running_loss / 100:.3f}")


            logger.print(f"epoch {epoch + 1}/{epochs} | average loss: {running_loss / len(trainloader):.3f}")
    
        logger.print("Finished distillation training")
        return student 


def test_model(model: nn.Module, testloader: DataLoader, device: torch.device, logger: object, model_config: dict) -> float:

    logger.print("Started testing")

    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device).long()
                
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

    logger.print("Finished testing")
    return accuracy
