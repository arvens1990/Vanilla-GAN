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
matplotlib.style.use('ggplot')

# learning parameters
batch_size = 512
epochs = 200
sample_size = 100 # fixed sample size
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
            Linear(self.nz + 10, 256),
            ReLU(),

            Linear(256, 512),
            ReLU(),

            Linear(512, 784),
            Tanh(),
        )

    def forward(self, x, k):
        x = torch.cat((x, k), dim=1)
        # print(x)
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = Sequential(
            Linear(self.n_input + 10, 1024),
            ReLU(),
            Linear(1024, 512),
            ReLU(),
            Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, x, k):
        x = x.view(-1, 784)
        x = torch.cat((x, k), dim=1)
        return self.main(x)


generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)
print('##### GENERATOR #####')
print(generator)
print('######################')
print('\n##### DISCRIMINATOR #####')
print(discriminator)
print('######################')

# optimizers

optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = nn.BCELoss()

# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake, onehot):
    b_size = data_real.size(0)
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)
    optimizer.zero_grad()
    output_real = discriminator(data_real, onehot)
    loss_real = criterion(output_real, real_label)
    output_fake = discriminator(data_fake, onehot)
    loss_fake = criterion(output_fake, fake_label)
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    return loss_real + loss_fake

# function to train the generator network
def train_generator(optimizer, data_fake, onehot):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)
    optimizer.zero_grad()
    output = discriminator(data_fake, onehot)
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    return loss

# create the noise vector
noise = create_noise(sample_size, nz)
onehot_out = np.array([[1 if j % 10 == i else 0 for i in range(10)] for j in range(sample_size)])
onehot_out = torch.from_numpy(onehot_out)
generator.train()
discriminator.train()

# path = "/content/drive/MyDrive/Deep_Learning/HW3/"
path = "./models/conditional/"
epochs = 200
# k = 10
length = 0.

losses_g = [] # to store generator loss after each epoch
losses_d = [] # to store discriminator loss after each epoch
images = [] # to store images generatd by the generator

for epoch in range(epochs):
    start = time.time()
    loss_g = 0.0
    loss_d = 0.0
    for bi, data in enumerate(data_loader):
        image, labels = data
        image = image.to(device)
        onehot = torch.zeros((labels.size(0), labels.max() + 1))
        onehot[np.arange(labels.size(0)), labels] = 1
        
        # print(image.shape, onehot.shape)
        # print(torch.cat())

        b_size = len(image)
        # run the discriminator for k number of steps
        for step in range(k):
            # print(create_noise(b_size, nz).shape)
            data_fake = generator(create_noise(b_size, nz), onehot).detach()
            data_real = image
            # train the discriminator network
            loss_d += train_discriminator(optim_d, data_real, data_fake, onehot)
        data_fake = generator(create_noise(b_size, nz), onehot)
        # train the generator network
        loss_g += train_generator(optim_g, data_fake, onehot)
    # create the final fake image for the epoch
    generated_img = generator(noise, onehot_out).cpu().detach()
    # make the images as grid
    generated_img = make_grid(generated_img, nrow = 10)
    # save the generated torch tensor models to disk
    save_generator_image(generated_img, path + f"gen_img{epoch}.png")
    images.append(generated_img)
    epoch_loss_g = loss_g / bi # total generator loss for the epoch
    epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)
    end = time.time() - start
    length += end
    mean_so_far = length / (epoch+1)
    time_left = (mean_so_far * (epochs - epoch - 1))/60
    
    print(f"Epoch {epoch} of {epochs}:\t\t{end:.2f} seconds;\ttotal: {length:.2f};\tminutes left: {time_left:.2f}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

n = 10

onehot = torch.zeros((n*3, n))

print(onehot)
onehot[np.arange(n*3), np.array([[i,i,i] for i in range(n)]).flatten()] = 1
print(onehot)


noise = create_noise(n*3, nz)

generated_img = generator(noise, onehot).cpu().detach()
# make the images as grid
generated_img = make_grid(generated_img, nrow = 3)
save_generator_image(generated_img, f"./models/conditional/conditional_generated_{n}.png")