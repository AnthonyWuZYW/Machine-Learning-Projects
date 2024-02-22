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
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

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

# Training 
def train(model, optimizer):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.view(-1, 28 * 28))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Testing 
def test(model):
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data

            outputs = model(inputs.view(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)

train(net, optimizer)
test(net)

# save the model
PATH = './model.pth'
torch.save(net.state_dict(), PATH)