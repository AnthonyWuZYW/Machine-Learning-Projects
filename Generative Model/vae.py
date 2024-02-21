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


# Implementation of Variational Autoencoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.input_size = 784
        self.hidden_size = 400

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(latent_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.input_size)

        self.mean = nn.Linear(self.hidden_size, latent_size)
        self.log_variance = nn.Linear(self.hidden_size, latent_size)

    def encode(self, x):
        # The encoder will take an input of size 784, and will produce two vectors of size latent_size (corresponding to the coordinatewise means and log_variances)
        # It should have a single hidden linear layer with 400 nodes using ReLU activations, and have two linear output layers (no activations)
        h = F.relu(self.linear1(x))
        means = self.mean(h)
        log_vars = self.log_variance(h)
        return means, log_vars

    def reparameterize(self, means, log_variances):
        # The reparameterization module lies between the encoder and the decoder
        # It takes in the coordinatewise means and log-variances from the encoder (each of dimension latent_size), and returns a sample from a Gaussian with the corresponding parameters
        sigma = torch.exp(log_variances * 0.5)
        epsilon = torch.randn_like(sigma)
        return means + sigma * epsilon

    def decode(self, z):
        # The decoder will take an input of size latent_size, and will produce an output of size 784
        # It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
        h = F.relu(self.linear2(z))
        return torch.sigmoid(self.linear3(h))

    def forward(self, x):
        # Apply the VAE encoder, reparameterization, and decoder to an input of size 784
        # Returns an output image of size 784, as well as the means and log_variances, each of size latent_size (they will be needed when computing the loss)
        means, log_vars = self.encode(x)
        z = self.reparameterize(means, log_vars)
        return self.decode(z), means, log_vars


def vae_loss_function(reconstructed_x, x, means, log_variances):
    # Compute the VAE loss
    # The loss is a sum of two terms: reconstruction error and KL divergence
    # Use cross entropy loss between x and reconstructed_x for the reconstruction error (as opposed to L2 loss as discussed in lecture -- this is sometimes done for data in [0,1] for easier optimization)
    # The KL divergence is -1/2 * sum(1 + log_variances - means^2 - exp(log_variances)) as described in lecture
    # Returns loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    reconstruction_loss = F.cross_entropy(reconstructed_x, x, reduction="sum")
    KL = -1/2 * torch.sum(1 + log_variances - means**2 - log_variances.exp())
    loss = reconstruction_loss + KL
    return loss, reconstruction_loss


def train(model, optimizer):
    # Trains the VAE for one epoch on the training dataset
    # Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)

    losses = 0.0
    reconstruction_losses = 0.0
    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to(device).view(-1, 784)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        reconstructed_x, means, log_vars = model(inputs)
        loss, reconstruction_loss = vae_loss_function(reconstructed_x, inputs, means, log_vars)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        reconstruction_losses += reconstruction_loss.item()

    avg_train_loss = losses / (i + 1) / batch_size
    avg_train_reconstruction_loss = reconstruction_losses / (i + 1) / batch_size
    return avg_train_loss, avg_train_reconstruction_loss


def test(model):
    # Runs the VAE on the test dataset
    # Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    losses = 0.0
    reconstruction_losses = 0.0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device).view(-1, 784)

        # test loss
        reconstructed_x, means, log_vars = model(inputs)
        loss, reconstruction_loss = vae_loss_function(reconstructed_x, inputs, means, log_vars)

        losses += loss.item()
        reconstruction_losses += reconstruction_loss.item()

    avg_test_loss = losses / (i + 1) / batch_size
    avg_test_reconstruction_loss = reconstruction_losses / (i + 1) / batch_size
    return avg_test_loss, avg_test_reconstruction_loss


epochs = 50
avg_train_losses = []
avg_train_reconstruction_losses = []
avg_test_losses = []
avg_test_reconstruction_losses = []

vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    avg_train_loss, avg_train_reconstruction_loss = train(vae_model, vae_optimizer)
    avg_test_loss, avg_test_reconstruction_loss = test(vae_model)

    avg_train_losses.append(avg_train_loss)
    avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
    avg_test_losses.append(avg_test_loss)
    avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(avg_train_reconstruction_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()

plt.plot(avg_test_reconstruction_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()
