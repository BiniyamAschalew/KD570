import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np

transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(checkpoint_path):
    model = models.resnet18(pretrained=False)  # Example model, change according to your needs
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform_pipeline(image).unsqueeze(0)

def process_activation(model, input_tensor, target_layer, iterations=30, lr=1.0):
    layer_output = {}

    def hook(module, input, output):
        layer_output[target_layer] = output

    handle = getattr(model, target_layer).register_forward_hook(hook)
    input_tensor.requires_grad = True

    for i in range(iterations):
        model.zero_grad()
        _ = model(input_tensor)
        activation = layer_output[target_layer].mean()
        activation.backward()
        input_tensor.data += lr * input_tensor.grad.data
        input_tensor.grad.zero_()

    handle.remove()
    return input_tensor.detach()

def deprocess_image(tensor):
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = tensor.clamp(0, 1)
    tensor = tensor.numpy().transpose(1, 2, 0)
    return (tensor * 255).astype(np.uint8)

def activation_maximization(checkpoint_path, image_path, target_layer):
    model = load_model(checkpoint_path)
    input_tensor = load_image(image_path)
    optimized_tensor = process_activation(model, input_tensor, target_layer)
    return deprocess_image(optimized_tensor)

# for usage
checkpoint_path = ''
image_path = ''
target_layer = ''  
resulting_image = activation_maximization(checkpoint_path, image_path, target_layer)
