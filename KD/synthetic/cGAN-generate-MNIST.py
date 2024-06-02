import os
import sys
import numpy as np
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim):
        super(Generator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.label_embedding = nn.Embedding(self.classes, self.classes)

        self.model = nn.Sequential(
            *self._create_layer(self.latent_dim + self.classes, 128, False),
            *self._create_layer(128, 256),
            *self._create_layer(256, 512),
            *self._create_layer(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def _create_layer(self, size_in, size_out, normalize=True):
        layers = [nn.Linear(size_in, size_out)]
        if normalize:
            layers.append(nn.BatchNorm1d(size_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, noise, labels):
        z = torch.cat((self.label_embedding(labels), noise), -1)
        x = self.model(z)
        x = x.view(x.size(0), *self.img_shape)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load the model and state_dict
model = Generator(10, 1, 28, 100).to(device)
model.load_state_dict(torch.load('cgan/netG.pth'))
model.load_state_dict(torch.load('cgan/netG.pth'))
model.eval()


print("generating synthetic MNIST images...")
num_samples = 120000
n_sample = 1000
num_sampling = int(num_samples/n_sample)
model.eval()
with torch.no_grad():
    for i in range(num_sampling):
        noise = torch.randn(n_sample, 100, device=device)
        labels = torch.randint(0, 10, (n_sample,), device=device)
        gen_imgs = model(noise, labels)
        gen_imgs = gen_imgs.view(n_sample, 1, 28, 28)
        x_all = torch.cat([gen_imgs])
        x_all = torch.clamp(x_all, -1, 1)
        np.savez(f'cgan/samples/{i+1}_of_{num_sampling}.npz', x_all.cpu().numpy())
        grid = make_grid(gen_imgs*-1 + 1, nrow=5)
        save_image(grid, f'cgan/samples/{i+1}_of_{num_sampling}.png')
        #save_image(grid,  f"/root/data/synthetic/MNIST/DDPM/{i+1}_of_{num_sampling}.png")
        print(f'sample_{i+1}_of_{num_sampling}.png saved'.format(i))
        
