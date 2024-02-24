from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms


batch_size = 100
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

criterion = nn.CrossEntropyLoss()

# perturbation budget
epsilon = 0.3
eta = 0.017
steps = 20

# Implementation of Projected Gradient Descent method
def PGD(model, images, labels, epsilon, eta):
    images = images.to(device)
    labels = labels.to(device)

    perturbed_image = images.clone()
    for i in range(steps):
        perturbed_image.requires_grad = True
        outputs = model(perturbed_image)

        model.zero_grad()
        loss = criterion(outputs, labels).to(device)
        loss.backward()

        perturbed_image = perturbed_image + eta * perturbed_image.grad.sign()
        eta = torch.clamp(perturbed_image - images, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(images + eta, min=0, max=1).detach_()
    return perturbed_image

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

def train(model, optimizer):
    for i, (data, target) in enumerate(train_loader, 0):
        model.zero_grad()
        perturbed_data = PGD(model, data, target, epsilon, eta)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(perturbed_data.view(-1, 28 * 28))
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()


net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(20):
    train(net, optimizer)
    print("Epoch" + str(1+epoch))

PATH = './model_adv.pth'
torch.save(net.state_dict(), PATH)
