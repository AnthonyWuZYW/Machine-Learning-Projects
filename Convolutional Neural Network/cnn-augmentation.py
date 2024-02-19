import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import statistics
import math

variance = 0.01
#variance = 0.1
#variance = 1

transform = transforms.Compose(
    [#torchvision.transforms.RandomHorizontalFlip(p=1), # Horizontal Flip
     #torchvision.transforms.RandomVerticalFlip(p=1), # Vertical Flip
     torchvision.transforms.Resize(32), transforms.ToTensor(),
     torchvision.transforms.Lambda(lambda x: x + math.sqrt(variance) * torch.randn_like(x)),  # blur
     transforms.Normalize((0.5), (0.5))])

batch_size = 64

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
print("Finish loading data")


# Implement VGG11
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)

        # Batch
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(128)
        self.batch3 = nn.BatchNorm2d(256)
        self.batch4 = nn.BatchNorm2d(512)

        # DropOut
        self.drop = nn.Dropout(0.5)

        # Pool
        self.pool = nn.MaxPool2d(2, 2)

        # FC
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.batch1(self.conv1(x))))
        x = self.pool(F.relu(self.batch2(self.conv2(x))))
        x = F.relu(self.batch3(self.conv3(x)))
        x = self.pool(F.relu(self.batch3(self.conv4(x))))
        x = F.relu(self.batch4(self.conv5(x)))
        x = self.pool(F.relu(self.batch4(self.conv6(x))))
        x = F.relu(self.batch4(self.conv6(x)))
        x = self.pool(F.relu(self.batch4(self.conv6(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# previous trained model
PATH = './trained-models/trained_epoch5.pth'

net = Net()
net.load_state_dict(torch.load(PATH))

criterion = nn.CrossEntropyLoss()

correct = 0
total = 0
losses = []

net.eval()

with torch.no_grad():
    running_loss = 0.0
    for i, data in enumerate(testloader, 0):
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)

        loss = criterion(outputs, labels)


        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()
        if i % 20 == 19:
            losses.append(running_loss / 20)
            running_loss = 0.0

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
print(statistics.mean(losses))


