from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

hidden_size = 400
input_size = 784


# Implementation of Generative Adversarial Network
class Generator(nn.Module):
    #The generator takes an input of size latent_size, and will produce an output of size 784.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        return torch.sigmoid(self.linear2(h))

class Discriminator(nn.Module):
    #The discriminator takes an input of size 784, and will produce an output of size 1.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its output
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(h))

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    #Trains both the generator and discriminator for one epoch on the training dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    discriminator_losses = 0.0
    generator_losses = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device).view(-1, 784)

        # train discriminator
        discriminator_optimizer.zero_grad()
        discriminator_output = discriminator(inputs)
        sample = torch.randn(batch_size, latent_size)
        discriminator_loss = F.binary_cross_entropy(discriminator_output, torch.ones(batch_size, 1)) + \
                             F.binary_cross_entropy(discriminator(generator(sample)), torch.zeros(batch_size, 1))
        discriminator_loss.backward()
        discriminator_optimizer.step()

        discriminator_losses += discriminator_loss.item()

        # train generator
        generator_optimizer.zero_grad()
        sample = torch.randn(batch_size, latent_size)
        generator_output = generator(sample)
        generator_loss = F.binary_cross_entropy(discriminator(generator_output), torch.ones(batch_size, 1))
        generator_loss.backward()
        generator_optimizer.step()
        generator_losses += generator_loss.item()

    avg_generator_loss = generator_losses / (i + 1) / batch_size
    avg_discriminator_loss = discriminator_losses / (i + 1) / batch_size
    return avg_generator_loss, avg_discriminator_loss

def test(generator, discriminator):
    #Runs both the generator and discriminator over the test dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    discriminator_losses = 0.0
    generator_losses = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device).view(-1, 784)

        # discriminator test loss
        discriminator_output = discriminator(inputs)
        sample = torch.randn(batch_size, latent_size)
        discriminator_loss = F.binary_cross_entropy(discriminator_output, torch.ones(batch_size, 1)) + \
                             F.binary_cross_entropy(discriminator(generator(sample)), torch.zeros(batch_size, 1))
        discriminator_losses += discriminator_loss.item()

        # generator test loss
        sample = torch.randn(batch_size, latent_size)
        generator_output = generator(sample)
        generator_loss = F.binary_cross_entropy(discriminator(generator_output), torch.ones(batch_size, 1))
        generator_losses += generator_loss.item()

    avg_generator_loss = generator_losses / (i + 1) / batch_size
    avg_discriminator_loss = discriminator_losses / (i + 1) / batch_size
    return avg_generator_loss, avg_discriminator_loss


epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()

plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()
