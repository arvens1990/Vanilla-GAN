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

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


MLP_network = MLP()
        

discriminator = MLP()
discriminator.load_state_dict(torch.load("./outputs/fmnist_classifier.pth", map_location=device))

optim_d = optim.Adam(discriminator.parameters(), lr=0.00005)

criterion = nn.CrossEntropyLoss()

losses = []
accuracies = []
epochs = 50

discriminator.train()

# function to train the discriminator network
def train_discriminator(optimizer, data, labels):
    optimizer.zero_grad()
    output = discriminator(data)
    # print(output.shape)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    return loss, output

for epoch in range(epochs):
    total = 0
    acc_loss = 0
    correct = 0
    for bi, (images, labels) in enumerate(data_loader):
        loss, output = train_discriminator(optim_d, images, labels)
        acc_loss += loss
        b_size = len(labels)
        total += b_size
        predicted = torch.argmax(output, dim=1)
        correct += (predicted==labels).sum()
        accuracy = correct/total
        avg_loss = loss/b_size
        if bi%20==0:
            print(f"Epoch {epoch}/{epochs}; Batch {bi}: Loss = {loss:.5f}\t\tAccuracy = {accuracy:.5f}")

    
    losses.append(acc_loss/total)
    accuracies.append(accuracy)


# load model and set to evaluate mode

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


generator_GAN = Generator(nz)
generator_WGAN = Generator(nz)
generator_unrolled = Generator(nz)

generator_GAN.load_state_dict(torch.load("./models/vanilla_gan/generator.pth", map_location=torch.device('cpu')))
generator_WGAN.load_state_dict(torch.load("./models/wasserstein/generator.pth", map_location=torch.device('cpu')))
generator_unrolled.load_state_dict(torch.load("./models/unrolled/generator.pt", map_location=torch.device('cpu')))
generator_GAN.eval()
generator_WGAN.eval()
generator_unrolled.eval()
# print(generator)

# generate 3,000 new images

sample_size = 3000

def create_noise(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)

nz = 128
# create noise

models = {'generator_GAN': generator_GAN, 'generator_WGAN': generator_WGAN, 'generator_unrolled': generator_unrolled}

for key, generator in models.items():

    noise = create_noise(sample_size, nz)

    # feed noise to generator
    new_images = generator(noise)

    # feed new images to discriminator
    new_softmax = discriminator(new_images)
    new_labels = torch.argmax(new_softmax, dim=1)

    plt.figure()
    plt.hist(new_labels.numpy(), bins=10)
    plt.title(key)
    plt.savefig(f"./outputs/new_labels_histogram_{key}.png")