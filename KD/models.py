from typing import List, Tuple
import os

import torch
import torch.nn as nn
import torchvision.models as models


def load_model(model_config: dict, device: torch.device, multiprocess: bool = False) -> Tuple[nn.Module, bool]:
    
    model = build_model(model_config, device, multiprocess)
    trained = False

    if model_config["load_from_path"]:
        print(f"Loading model from {model_config['model_path']}")

        if os.path.exists(model_config["model_path"]):
            model.load_state_dict(torch.load(model_config["model_path"]))
            trained = True
            print(f"Model loaded from {model_config['model_path']}")

        else:
            print(f"Model path {model_config['model_path']} does not exist. Using built model instead")

    return model, trained



# def build_model(model_name: str, config: dict, input_dict: dict = None ) -> nn.Module:
def build_model(model_config: dict, device: torch.device, multiprocess: bool = False) -> nn.Module:
        
        out_channels = model_config["num_classes"]
        # print(type(out_channels))
        # assert(0)
        pretrained = model_config["pretrained"]
        model_name = model_config["model"].lower()

        shape = model_config["input_shape"]
        input_channels = shape[0]
        input_size = shape[1]
    

        if model_name == "custom":
            model = ConvNetBuilder(input_channels, out_channels, model_config, input_size)

        
        elif model_name == "resnet18":
            model = ResNet18Builder(input_channels, out_channels, pretrained)

        elif model_name == "resnet50":
            model = ResNet50Builder(input_channels, out_channels, pretrained)

        elif model_name == "resnet101":
            model = ResNet101Builder(input_channels, out_channels, pretrained)

        elif model_name == "vgg16":
            model = VGG16Builder(input_channels, out_channels, pretrained)

        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        # if multiprocess:
        #     model = nn.DataParallel(model)

        # model = model.to(device)
        print(f"Model build as {model_name}, pretrained: {pretrained}, output channels: {out_channels}")
        return model



class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dropout: float = 0.5):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.dropout(self.relu(self.bn(self.conv(x)))))
    
class ConvNetBuilder(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, input_dict: dict, input_size: int = 28):
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

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        for layer in range(self.num_layers):
            x = self.conv_blocks[layer](x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if not hard:
            return x

        return x 
    

class ResNet18Builder(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, pretrained: bool = True):
        super(ResNet18Builder, self).__init__()

        self.resnet18 = models.resnet18(pretrained=pretrained)

        if input_channels != 3:
            self.resnet18.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if output_channels != 1000:
            self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet18(x)
    
class ResNet50Builder(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, pretrained: bool = True):
        super(ResNet50Builder, self).__init__()

        self.resnet50 = models.resnet50(pretrained=pretrained)

        if input_channels != 3:
            self.resnet50.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if output_channels != 1000:
            self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet50(x)
    
class ResNet101Builder(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, pretrained: bool = True):
        super(ResNet101Builder, self).__init__()

        self.resnet101 = models.resnet101(pretrained=pretrained)

        if input_channels != 3:
            self.resnet101.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if output_channels != 1000:
            self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet101(x)
    
class VGG16Builder(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, pretrained: bool = True):
        super(VGG16Builder, self).__init__()

        self.vgg16 = models.vgg16(pretrained=pretrained)

        if input_channels != 3:
            self.vgg16.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)

        if output_channels != 1000:
            self.vgg16.classifier[6] = nn.Linear(self.vgg16.classifier[6].in_features, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg16(x)

    




