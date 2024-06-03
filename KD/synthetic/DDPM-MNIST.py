from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import numpy as np
from tqdm import tqdm

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        # x is (noisy) image, t is timestep,
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed  time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down2)  # add embeddings
        up3 = self.up2(up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

def forward_process(x):
    """
    this method is used in training, so samples t and noise randomly
    """
    _ts = torch.randint(1, T+1, (x.shape[0],)).to(x.device)  # t ~ Uniform(0, n_T)
    noise = torch.randn_like(x)  # eps ~ N(0, 1)

    sqrtab_ = torch.randn_like(x)
    sqrtmab_ = torch.randn_like(x)
    for i in range(len(_ts)):
        sqrtab_[i][:][:][:]  = sqrtab[_ts[i]]
        sqrtmab_[i][:][:][:] = sqrtmab[_ts[i]]
    x_t = sqrtab_ * x + sqrtmab_ * noise   # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps

    return x_t, noise, _ts
        
def sampling(eps_model, n_sample, size, T, device):
    x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
    x_i_store = [] # keep track of generated steps in case want to plot something

    for i in tqdm(range(T, 0, -1)):
        # print(f'sampling timestep {i}',end='\r')
        t_is = torch.tensor([i]).to(device)
        t_is = t_is.repeat(n_sample, 1, 1, 1)
        z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

        eps = eps_model(x_i, t_is/T)
        alpha_t_ = alpha_t[t_is]
        sqrtmab_ = sqrtmab[t_is]
        sqrt_beta_t_ = sqrt_beta_t[t_is]
        x_i = 1/torch.sqrt(alpha_t_) * (x_i - (1-alpha_t_)/sqrtmab_ * eps) + sqrt_beta_t_ * z
        if i%20==0 or i==T or i<8:
            x_i_store.append(x_i.detach().cpu().numpy())

    x_i_store = np.array(x_i_store)
    return x_i, x_i_store

if __name__ == '__main__':
    beta1 = 1e-4
    beta2 = 0.02
    T = 500

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    beta_t = beta_t.cuda()
    print(beta_t)

    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t

    alpha_t_bar = torch.randn_like(beta_t)
    for i in range(len(alpha_t_bar)):
        alpha_t_bar[i] = torch.prod(alpha_t[:i+1])
    sqrtab = torch.sqrt(alpha_t_bar)        # \sqrt{\bar{\alpha}_t}
    sqrtmab = torch.sqrt(1-alpha_t_bar)     # \sqrt{1-\bar{\alpha}_t}

    n_epoch = 500
    batch_size = 128
    device = "cuda:0"
    n_classes = 10
    n_feat = 128
    lrate = 1e-4

    epsilon_model = ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes)
    epsilon_model.to(device)

    # normalize values to -1 to 1
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # mnist is already normalised 0 to 1

    train_dataset = MNIST("../data", train=True, download=True, transform=tf)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(epsilon_model.parameters(), lr=lrate)
    loss_mse = nn.MSELoss()

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        epsilon_model.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x_0, target in pbar:
            optim.zero_grad()
            x_0 = x_0.to(device)

            x_t, noise, _ts = forward_process(x_0)
            eps_theta = epsilon_model(x_t, _ts / T)
            loss = loss_mse(eps_theta, noise)
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

    torch.save(epsilon_model.state_dict(), f'../trained_models/MNIST-DDPM-epoch-{n_epoch}.pth')

    # for eval, save an image of currently generated samples (top rows)
    # followed by real images (bottom rows)
    print("generating synthetic MNIST images...")
    num_samples = 120000
    n_sample = 1000
    num_sampling = int(num_samples/n_sample)
    epsilon_model.eval()
    with torch.no_grad():
        for i in range(num_sampling):
            x_gen, x_gen_store = sampling(epsilon_model, n_sample, (1, 28, 28), T, device)
            x_all = torch.cat([x_gen])
            x_all = torch.clamp(x_all, min=-1.00, max=1.00)
            np.savez(f'../data/synthetic/MNIST/DDPM/{i+1}_of_{num_sampling}.npz', data=x_all.cpu().detach().numpy())
            grid = make_grid(x_all*-1 + 1, nrow=25)
            save_image(grid,  f"../data/synthetic/MNIST/DDPM/{i+1}_of_{num_sampling}.png")
            print('saved image at ' + f"../data/synthetic/MNIST/DDPM/{i+1}_of_{num_sampling}.npz")
