import torch
import numpy as np
import random
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision import transforms

def get_loader(batch_size):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform_svhn  = transforms.Compose([transforms.Resize(mnist_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_mnist = transforms.Compose([transforms.Resize(svhn_size), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    
    svhn = datasets.SVHN(root=svhn_path, download=download, transform=transform_svhn)
    mnist = datasets.MNIST(root=mnist_path, download=download, transform=transform_mnist)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mnist_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return svhn_loader, mnist_loader

class Residual_Block(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_dim)
        self.act_fn1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.act_fn2 = nn.ReLU()

    def forward(self, x):
        conv = self.bn2(self.conv2(self.act_fn1(self.bn1(self.conv1(x)))))  # from Conv2d(in_dim->mid_dim) to BatchNorm2d(out_dim)
        sum = x + conv                                                      # adding skip connection
        out = self.act_fn2(sum)                                             # final ReLU
        return out
    
class Generator(nn.Module):
    def __init__(self, type, conv_dim=64):
        super(Generator, self).__init__()
        if type == 'mnist_to_svhn':
            in_dim, out_dim = 1, 3
        elif type == 'svhn_to_mnist':
            in_dim, out_dim = 3, 1
        # encoding blocks
        self.conv1 = nn.Conv2d(in_dim, conv_dim, kernel_size=3, padding = 1)
        self.bn1 = nn.BatchNorm2d(conv_dim)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, padding = 1)
        self.bn2 = nn.BatchNorm2d(conv_dim*2)
        # residual blocks
        self.res1 = Residual_Block(conv_dim*2, conv_dim, conv_dim*2)
        self.res2 = Residual_Block(conv_dim*2, conv_dim, conv_dim*2)
        # decoding blocks
        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim, kernel_size=3, padding = 1)
        self.bn3 = nn.BatchNorm2d(conv_dim)
        self.conv4 = nn.Conv2d(conv_dim, out_dim, kernel_size=3, padding = 1)
        # other layers
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.downsample = nn.MaxPool2d(kernel_size=3, padding=1, stride = 2)
        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear')

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.downsample(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.downsample(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.upsample(out)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.upsample(out)
        out = self.tanh(self.conv4(out))
        return out
    
class Discriminator(nn.Module):
    def __init__(self, type, conv_dim=64):
        super(Discriminator, self).__init__()
        if type == 'mnist':
            in_dim = 1
        elif type == 'svhn':
            in_dim = 3
        # in_dim, out_dim = 1, 1

        # learnable layers
        self.conv1 = nn.Conv2d(in_dim, conv_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_dim)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_dim * 2)
        self.conv3 = nn.Conv2d(conv_dim * 2, conv_dim * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_dim * 2)
        self.conv4 = nn.Conv2d(conv_dim * 2, conv_dim * 2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_dim * 2)
        
        self.fc = nn.Linear(conv_dim * 2, 11) # Note that classes 0 through 9 are for real images and class 10 is for fake images.

        # other functions
        self.relu = nn.ReLU()
        self.downsample = nn.MaxPool2d(kernel_size=3, padding= 1, stride=2)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.downsample(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.downsample(out)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.downsample(out)
        out = self.relu(self.bn4(self.conv4(out)))
        out = out.flatten(2,3).max(-1)[0]
        out = self.fc(out)
        return out
    
def optimize_model(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def show_log(step, train_iters, D_ms_losses, D_sm_losses, G_msm_losses, G_ses_losses):
    D_ms_avg = np.array(D_ms_losses).mean()
    D_sm_avg = np.array(D_sm_losses).mean()
    G_msm_avg = np.array(G_msm_losses).mean()
    G_sms_avg = np.array(G_ses_losses).mean()

    print('Step [%d/%d], D_ms_loss: %.4f, D_sm_loss: %.4f, G_msm_loss: %.4f, G_sms_loss: %.4f'
    % (step + 1, train_iters, D_ms_avg, D_sm_avg, G_msm_avg, G_sms_avg))
    
def save_img(fixed_mnist, fixed_svhn, G, F, idx, total):
    fake_svhn = G(fixed_mnist)
    fake_mnist = F(fixed_svhn)
    
    np.savez(f'../data/synthetic/MNIST/CycleGAN/{idx}_of_{total}.npz', data=fake_mnist.cpu().detach().numpy())
    np.savez(f'../data/synthetic/SVHN/CycleGAN/{idx}_of_{total}.npz', data=fake_svhn.cpu().detach().numpy())    
    print('saved image at ' + f"../data/synthetic/(MNIST, SVHN)/CycleGAN/{idx}_of_{total}.npz")

def Train_D_ms(Dx, Dy, G, m_data, m_label):
    '''
    # parameter:
    Dx, Dy: Discriminator for source domain(X, MNIST) and target domain(Y, SVHN), respectively.
    G: Generator which transfers X to Y
    m_data: source domain data (i.e., MNIST real images)
    m_label: source domain label (from 0 to 9)

    Train two discriminators(Dx, Dy) in mnist-svhn cycle. 
    Please refer to the above figure that represents the path from X to Y and equation (1),(3).
    Unlike the conventional GAN, here, the discriminator classifies 11 classes. You have to use 1) the original labels(from 0 to 9) when training the discriminator with real images, and 2) the fake label(10) when training the discriminator with fake(generated) images. You should make the fake label for training.
    So, "torch.nn.functional.cross_entropy()" will be used for calculating losses. 

    
    Return:
    D_ms_loss.item(): item of the loss
    '''    

    # train mnist-svhn cycle

    # 1) Compute the loss of Dx (Dx_loss) with real images
    real_validity = Dx(m_data)                                          # one-hot encoding of the label from MNIST discriminator of real MNIST images 
    Dx_loss = torch.nn.functional.cross_entropy(real_validity, m_label) # the CE loss of predicted labels w.r.t. ground-truth class label

    # 2) Compute the loss of Dy (Dy_loss) with fake images
    fake_images = G(m_data)                                                                               # generate SVHN images from MNIST images
    fake_labels_output = torch.Tensor([10 for i in range(len(m_data))]).type(torch.LongTensor).to(device) # fake images so the ground-truth label is 10
    fake_validity = Dy(fake_images)                                               # corresponding one-hot encoding of the labels from SVHN discriminator
    Dy_loss = torch.nn.functional.cross_entropy(fake_validity, fake_labels_output)                        # the CE loss w.r.t. ground-truth fake(10) labels

    D_ms_loss = Dx_loss + Dy_loss
    optimize_model(disOptimizer, D_ms_loss)
    return D_ms_loss.item()

def Train_D_sm(Dx, Dy, F, s_data, s_label):
    '''
    # parameter:
    Dx, Dy: Discriminator for source domain(X, MNIST) and target domain(Y, SVHN), respectively.
    F: Generator which transfers Y to X
    s_data: target domain data (i.e., SVHN real images)
    s_label: target domain label (from 0 to 9)

    Train discriminators in svhn-mnist cycle. 
    Please refer to the above figure that represents the path from Y to X and equation (1),(3).
    Unlike the conventional GAN, here, the discriminator classifies 11 classes. You have to use 1) the original labels(from 0 to 9) when training the discriminator with real images, and 2) the fake label(10) when training the discriminator with fake(generated) images. You should make the fake label for training.
    So, "torch.nn.functional.cross_entropy()" will be used for calculating losses. 

    Return:
    D_sm_loss.item(): item of the loss
    '''   

    # train svhn-mnist cycle

    # 1) Compute the loss of Dy (Dy_loss) with real images
    real_validity = Dy(s_data)                                          # one-hot encoding of the label from MNIST discriminator of real SVHN images 
    Dy_loss = torch.nn.functional.cross_entropy(real_validity, s_label) # the CE loss of predicted labels w.r.t. ground-truth class label

    # 2) Compute the loss of Dx (Dx_loss) with fake images
    fake_images = F(s_data)                                                                               # generate MNIST images from SVHN images
    fake_labels_output = torch.Tensor([10 for i in range(len(s_data))]).type(torch.LongTensor).to(device) # fake images so the ground-truth label is 10
    fake_validity = Dx(fake_images)                                               # corresponding one-hot encoding of the labels from MNIST discriminator
    Dx_loss = torch.nn.functional.cross_entropy(fake_validity, fake_labels_output)                        # the CE loss w.r.t. ground-truth fake(10) labels 

    D_sm_loss = Dx_loss + Dy_loss
    optimize_model(disOptimizer, D_sm_loss)
    return D_sm_loss.item()

# train mnist-svhn-mnist cycle
def Train_G_msm(Dy, G, F, m_data, m_label):
    '''
    # parameter:
    Dy: Discriminator for target domain(Y, SVHN).
    G, F: Generator for X to Y and Y to X.
    m_data: source domain data (i.e., MNIST real images)
    m_label: source domain label (from 0 to 9)

    Train mnist-svhn-mnist cycle. 
    Please refer to the above figure that represents the path from X to Y to X and equation (1),(5).
    Note that you have to use original labels(from 0 to 9) when training the generator. 
    So, "torch.nn.functional.cross_entropy()" will be used for calculating losses. 
    Also, this function has to include cycle consistency loss(i.e. Lcyc) with mean squared error loss.

    Return:
    G_msm_loss.item(): item of the loss
    '''

    # Compute the generator G loss (G_loss) with fake images
    fake_images = G(m_data)                                             # generate SVHN images from MNIST images
    fake_validity = Dy(fake_images)                                     # corresponding one-hot encoding of the label from SVHN discriminator 
    G_loss = torch.nn.functional.cross_entropy(fake_validity, m_label)  # the CE loss of predicted labels w.r.t. ground-truth class labels

    # Compute the cycle consistency loss (Lcyc)
    cycle_images = F(fake_images)                                       # regenerate MNIST images from the previously generated SVHN images
    Lcyc = torch.nn.functional.mse_loss(m_data, cycle_images)           # the MSE loss of regenerated MNIST images w.r.t. original MNIST images

    G_msm_loss = G_loss + 100 * Lcyc
    optimize_model(genOptimizer, G_msm_loss)
    return G_msm_loss.item()

# train svhn-mnist-svhn cycle
def Train_G_sms(Dx, G, F, s_data, s_label):
    '''
    # parameter:
    Dx: Discriminator for source domain(X, MNIST).
    G, F: Generator for X to Y and Y to X.
    s_data: target domain data (i.e., SVHN real images)
    s_label: target domain label (from 0 to 9)

    Train svhn-mnist-svhn cycle. 
    Please refer to the above figure that represents the path from Y to X to Y and equation (3),(5).
    Note that you have to use original labels(from 0 to 9) when training the generator.
    So, "torch.nn.functional.cross_entropy()" will be used for calculating losses. 
    Also, this function has to include cycle consistency loss(i.e. Lcyc) with mean squared error loss.

    Return:
    G_sms_loss.item(): item of the loss
    ''' 

    # Compute the generator F loss (F_loss) with fake images
    fake_images = F(s_data)                                             # generate MNIST images from SVHN images
    fake_validity = Dx(fake_images)                                     # corresponding one-hot encoding of the label from MNIST discriminator 
    F_loss = torch.nn.functional.cross_entropy(fake_validity, s_label)  # the CE loss of predicted labels w.r.t. ground-truth class labels

    # Compute the cycle consistency loss (Lcyc)
    cycle_images = G(fake_images)                                       # regenerate SVHN images from the previously generated MNIST images
    Lcyc = torch.nn.functional.mse_loss(s_data, cycle_images)           # the MSE loss of regenerated SVHN images w.r.t. original SVHN images

    G_sms_loss = F_loss + 100 * Lcyc
    optimize_model(genOptimizer, G_sms_loss)
    return G_sms_loss.item()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # model hyper-parameters
    svhn_size=32
    mnist_size=28
    g_conv_dim=64
    d_conv_dim=64
    use_labels=True
    num_classes=10

    # training hyper-parameters
    train_iters=2000
    batch_size=128
    num_workers=2

    # misc
    mode='train'
    model_path='../models'
    mnist_path='../data'
    svhn_path='../data/SVHN'
    download=True
        
    svhn_loader, mnist_loader = get_loader(batch_size)
    svhn_iter = iter(svhn_loader)
    mnist_iter = iter(mnist_loader)
    iter_per_epoch = min(len(svhn_iter), len(mnist_iter))

    # Build models and their optimizers
    # Generator
    G = Generator(conv_dim=64, type='mnist_to_svhn').cuda()
    F = Generator(conv_dim=64, type='svhn_to_mnist').cuda()

    # Discriminator
    Dx = Discriminator(conv_dim=64, type='mnist').cuda()
    Dy = Discriminator(conv_dim=64, type='svhn').cuda()

    genParams = list(G.parameters()) + list(F.parameters())
    disParams = list(Dx.parameters()) + list(Dy.parameters())

    genOptimizer = optim.Adam(genParams)
    disOptimizer = optim.Adam(disParams)

    D_ms_losses, D_sm_losses, G_msm_losses, G_sms_losses = [], [], [], []

    for step in range(train_iters + 1):
        # reset data_iter for each epoch
        if (step + 1) % iter_per_epoch == 0:
            mnist_iter = iter(mnist_loader)
            svhn_iter = iter(svhn_loader)

        # load svhn and mnist dataset
        s_data, s_label = next(svhn_iter)
        s_data, s_label = s_data.cuda(), s_label.cuda().long().squeeze()
        m_data, m_label = next(mnist_iter)
        m_data, m_label = m_data.cuda(), m_label.cuda().long().squeeze() # modified
        
        D_ms_losses.append(Train_D_ms(Dx, Dy, G, m_data, m_label))
        D_sm_losses.append(Train_D_sm(Dx, Dy, F, s_data, s_label))
        G_msm_losses.append(Train_G_msm(Dy, G, F, m_data, m_label))
        G_sms_losses.append(Train_G_sms(Dx, G, F, s_data, s_label))

        # print the log info
        if (step + 1) % 100 == 0:
            show_log(step, train_iters, D_ms_losses, D_sm_losses, G_msm_losses, G_sms_losses)
    
    torch.save(G.state_dict(), f'../models/MNIST-SVHN-CycleGAN-G-epoch-{train_iters}.pth')
    torch.save(F.state_dict(), f'../models/MNIST-SVHN-CycleGAN-F-epoch-{train_iters}.pth')
    torch.save(Dx.state_dict(), f'../models/MNIST-SVHN-CycleGAN-Dx-epoch-{train_iters}.pth')
    torch.save(Dy.state_dict(), f'../models/MNIST-SVHN-CycleGAN-Dy-epoch-{train_iters}.pth')
    
    num_samples = 120000
    n_sample = 1000
    num_sampling = int(num_samples/n_sample)
    
    svhn_loader, mnist_loader = get_loader(n_sample)
    svhn_iter = iter(svhn_loader)
    mnist_iter = iter(mnist_loader)
    iter_per_epoch = min(len(svhn_iter), len(mnist_iter))
    
    for idx in range(num_sampling):
        # reset data_iter for each epoch
        if (idx + 1) % iter_per_epoch == 0:
            mnist_iter = iter(mnist_loader)
            svhn_iter = iter(svhn_loader)

        # load svhn and mnist dataset
        s_data, _ = next(svhn_iter)
        s_data = s_data.cuda()
        m_data, _ = next(mnist_iter)
        m_data = m_data.cuda()
        
        save_img(m_data, s_data, G, F, idx+1, num_sampling)
        