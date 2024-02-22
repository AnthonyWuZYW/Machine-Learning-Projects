from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torchvision import datasets, transforms


batch_size = 100
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

# perturbation budget
epsilon = 0.1
criterion = nn.CrossEntropyLoss()


# Implemention of a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

PATH = './model.pth'
net = Net()
net.load_state_dict(torch.load(PATH))


# Implementation of Fast Gradient Sign Method
def FGSM(image, epsilon, grad):
    signed_grad = grad.sign()
    perturbed_image = image + epsilon*signed_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# Generate adversarial examples and test it on trained model
def generate(model):
    total = 0
    correct = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data.view(-1, 28 * 28))
        _, init_pred = torch.max(output.data, 1)

        # Calculate the loss
        loss = criterion(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = FGSM(data, epsilon, data_grad)

        # test on adversarial images
        _, final_pred = torch.max(model(perturbed_data).data, 1)
        total += target.size(0)
        correct += (final_pred == target).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

generate(net)

