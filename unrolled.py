from my_functions import *
import numpy as np
import pandas as pd
import torch
import torchvision as tv
import re
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Module, GRU, Embedding, Linear, Sigmoid, CrossEntropyLoss, ReLU, Tanh, Sequential
from torch import nn
from torchvision import transforms
import torch.optim as optim
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
matplotlib.style.use('ggplot')

# learning parameters
batch_size = 32
epochs = 200
sample_size = 64 # fixed sample size
nz = 128 # latent vector size
k = 1 # number of steps to apply to the discriminator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
])
to_pil_image = transforms.ToPILImage()

fmnist = datasets.FashionMNIST(root='./', train=True, download=True, transform=transform)
data_loader = DataLoader(fmnist, batch_size=batch_size, shuffle=True)

class Generator(Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = Sequential(
            Linear(self.nz, 256),
            ReLU(),

            Linear(256, 512),
            ReLU(),

            Linear(512, 784),
            Tanh(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = Sequential(
            Linear(self.n_input, 1024),
            ReLU(),
            Linear(1024, 512),
            ReLU(),
            Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()    

generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)
print('##### GENERATOR #####')
print(generator)
print('######################')
print('\n##### DISCRIMINATOR #####')
print(discriminator)
print('######################')

# loss function
criterion = nn.BCELoss()

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)


# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake, discriminator, create_graph=False):
    b_size = data_real.size(0)
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)
    optimizer.zero_grad()
    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)
    # loss_real.backward()
    # loss_fake.backward()
    loss = loss_fake + loss_real
    loss.backward(create_graph=create_graph)
    optimizer.step()
    return loss.item()

# function to train the generator network
def train_generator(optimizer, data_fake, discriminator):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)
    optimizer.zero_grad()
    output = discriminator(data_fake)
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    return loss.item()

    # create the noise vector
noise = create_noise(sample_size, nz)
generator.train()
discriminator.train()

# path = "/content/drive/MyDrive/Deep_Learning/HW3/outputs_unrolled_gan/"
path = "./models/unrolled/"
epochs = 200
k = 1

losses_g = [] # to store generator loss after each epoch
losses_d = [] # to store discriminator loss after each epoch
losses_ud = []
images = [] # to store images generatd by the generator
length = 0.

for epoch in range(epochs):
    start = time.time()
    # length = 0.
    loss_g = 0.0
    loss_d = 0.0
    loss_ud = 0.
    for bi, data in enumerate(data_loader):
        image, _ = data
        image = image.to(device)
        b_size = len(image)
        
        
        # print(create_noise(b_size, nz).shape)
        data_fake = generator(create_noise(b_size, nz)).detach()
        data_real = image
        # train the discriminator network
        loss_d += train_discriminator(optim_d, data_real, data_fake, discriminator)
        
        # unroll
        backup = copy.deepcopy(discriminator)
        # run the unrolled discriminator for k number of steps
        for step in range(k):
            data_fake = generator(create_noise(b_size, nz)).detach()
            loss_ud += train_discriminator(optim_d, data_real, data_fake, discriminator, create_graph=True)
            
        # data_fake = generator(create_noise(b_size, nz))
        # train the generator network
        data_fake = generator(create_noise(b_size, nz)).detach()
        loss_g += train_generator(optim_g, data_fake, discriminator)
        discriminator.load(backup)
        del backup

        
    # create the final fake image for the epoch
    if epoch%1==0:
        generated_img = generator(noise).cpu().detach()
        # make the images as grid
        generated_img = make_grid(generated_img)
        # save the generated torch tensor models to disk
        save_generator_image(generated_img, path + f"gen_img{epoch}.png")
    images.append(generated_img)
    epoch_loss_g = loss_g / bi # total generator loss for the epoch
    epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
    epoch_loss_ud = loss_ud / (bi*k)
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)
    losses_ud.append(epoch_loss_ud)
    end = time.time() - start
    length += end
    mean_so_far = length / (epoch+1)
    time_left = (mean_so_far * (epochs - epoch - 1))/60
    
    print(f"Epoch {epoch} of {epochs}:\t\t{end:.2f} seconds;\ttotal: {length:.2f};\tminutes left: {time_left:.2f}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.5f}, Unrolled Discriminator loss: {epoch_loss_ud:.5f}")